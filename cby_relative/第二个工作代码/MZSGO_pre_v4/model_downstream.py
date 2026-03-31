"""
Downstream task model (MZSGO-DA) -- End-to-End dual Adapter + concatenation fusion strategy.

Core design:
1. No static embeddings, end-to-end training
2. Dual Adapter strategy:
   - adapter_0 (pretrained) → frozen, preserves domain functional semantic prior
   - adapter_1 (newly added for downstream) → trainable, learns task-specific adaptation
3. ESM2 + adapter_0 + adapter_1 all participate in forward computation
4. Sequence features and GO label text embeddings separately projected then concatenated,
   fed into classification MLP for function prediction probabilities

Fusion method (compared to Chapter 3 MZSGO's gated fusion):
- This chapter only involves sequence and label two-way inputs
- Sequence representation already injected with domain functional prior via adapter pretraining
- Information fusion complexity is lower, concatenation suffices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

from esm_adapter import (
    inject_adapters_into_esm2,
    freeze_esm_parameters,
    freeze_adapter_group,
    load_adapter_state_dict,
)


class FeatureDropout(nn.Module):
    """
    Modality dropout strategy: apply dropout to sequence features during training
    to enhance robustness to sequence-side noise and missing perturbations.
    """
    def __init__(self, dropout_prob=0.15):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def forward(self, seq_feat, label_feat):
        if not self.training:
            return seq_feat, label_feat
        
        batch_size = seq_feat.size(0)
        device = seq_feat.device
        # Modality-level random dropout for sequence features
        mask = (torch.rand(batch_size, 1, device=device) > self.dropout_prob).float()
        seq_feat = seq_feat * mask
        
        return seq_feat, label_feat


class EndToEndMZSGO(nn.Module):
    """
    End-to-End MZSGO-DA downstream task model.
    
    Architecture:
    - ESM2 + adapter_0(frozen) + adapter_1(trainable) → sequence representation h_seq ∈ R^1280
    - h_seq → W_seq → LN → GELU → Dropout → h_seq' ∈ R^512
    - E_label → W_label → LN → GELU → Dropout → h_label' ∈ R^512
    - Modality dropout (only on h_seq', p=0.15)
    - H_fused = [h_seq' ∥ h_label'] ∈ R^1024
    - ŷ = σ(W_out · Dropout(GELU(LN(W_cls · H_fused))))
    """
    def __init__(self, esm_type='esm2_t33_650M_UR50D',
                 nlp_dim=2560, hidden_dim=512, dropout=0.3,
                 num_adapter_layers=16, bottleneck_dim=None,
                 adapter_dropout=0.0,
                 pretrain_adapter_ckpt=None,
                 feature_dropout_prob=0.15):
        super().__init__()
        
        # === ESM2 + Dual Adapter ===
        if esm_type == 'esm2_t33_650M_UR50D':
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.num_layers = 33
            self.embed_dim = 1280
        elif esm_type == 'esm2_t36_3B_UR50D':
            self.esm_model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
            self.num_layers = 36
            self.embed_dim = 2560
        else:
            self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.num_layers = 33
            self.embed_dim = 1280
        
        if bottleneck_dim is None:
            bottleneck_dim = self.embed_dim // 2
        
        # Inject dual Adapter: adapter_0(pretrained) + adapter_1(downstream)
        self.adapter_params = inject_adapters_into_esm2(
            self.esm_model,
            num_adapter_layers=[num_adapter_layers, num_adapter_layers],
            adapter_names=['adapter_0', 'adapter_1'],
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_dropout=adapter_dropout,
        )
        
        # Freeze ESM2 original parameters
        freeze_esm_parameters(self.esm_model, self.adapter_params)
        
        # Load pretrained adapter_0 and freeze
        if pretrain_adapter_ckpt is not None:
            self._load_pretrain_adapter(pretrain_adapter_ckpt)
        
        # Freeze adapter_0, only train adapter_1
        freeze_adapter_group(self.adapter_params, 'adapter_0', freeze=True)
        freeze_adapter_group(self.adapter_params, 'adapter_1', freeze=False)
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # === Feature projection layers ===
        # Sequence feature projection: h_seq ∈ R^1280 → h_seq' ∈ R^512
        self.seq_proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # GO label text embedding projection: E_label ∈ R^nlp_dim → h_label' ∈ R^512
        self.nlp_proj = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Modality dropout (only on sequence features, p=0.15)
        self.feature_dropout = FeatureDropout(dropout_prob=feature_dropout_prob)
        
        # Concatenation fusion + classification MLP
        # H_fused = [h_seq' ∥ h_label'] ∈ R^1024 → MLP → 1
        fused_dim = hidden_dim * 2  # 512 + 512 = 1024
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def _load_pretrain_adapter(self, ckpt_path):
        """
        Load pretrained adapter_0 parameters.
        
        Pretrained checkpoint adapter parameter names look like:
            layer_17.adapter_0.0.module.0.weight  (ResMLP after attention)
            layer_17.adapter_0.1.module.0.weight  (ResMLP after FFN)
        
        Downstream model has same naming (because adapter_names=['adapter_0', 'adapter_1']),
        so load with strict=False (adapter_1 parameters not in checkpoint, remain randomly initialized).
        """
        import os
        if not os.path.exists(ckpt_path):
            print(f"WARNING: Pretrain adapter checkpoint not found: {ckpt_path}")
            return
        
        print(f"Loading pretrained adapter from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'adapter_state_dict' in checkpoint:
            adapter_state = checkpoint['adapter_state_dict']
        elif 'model_state_dict' in checkpoint:
            # Extract adapter parameters from full checkpoint
            adapter_state = {}
            for key, value in checkpoint['model_state_dict'].items():
                if 'adapter_params.' in key:
                    new_key = key.split('adapter_params.')[-1]
                    adapter_state[new_key] = value
        else:
            adapter_state = checkpoint
        
        missing, unexpected = load_adapter_state_dict(
            self.adapter_params, adapter_state, strict=False
        )
        print(f"  Loaded adapter_0 weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            # Expect adapter_1 parameters to be missing (they are newly added)
            adapter_1_missing = [k for k in missing if 'adapter_1' in k]
            other_missing = [k for k in missing if 'adapter_1' not in k]
            print(f"  adapter_1 keys (expected missing): {len(adapter_1_missing)}")
            if other_missing:
                print(f"  WARNING: Other missing keys: {other_missing}")
    
    def forward_esm(self, tokens):
        """
        Get sequence representation via ESM2 + dual Adapter.
        
        Manually implement forward to better control memory (skip LM Head etc.).
        """
        assert tokens.ndim == 2
        
        model = self.esm_model
        padding_mask = tokens.eq(model.padding_idx)
        
        x = model.embed_scale * model.embed_tokens(tokens)
        
        if getattr(model, 'token_dropout', False):
            x.masked_fill_((tokens == model.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (tokens == model.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        
        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)
        
        if not padding_mask.any():
            padding_mask_for_layers = None
        else:
            padding_mask_for_layers = padding_mask
        
        # Iterate through all layers (already replaced with AdapterTransformerLayer)
        for layer in model.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask_for_layers,
                        need_head_weights=False)
        
        # emb_layer_norm_after
        if hasattr(model, 'emb_layer_norm_after') and model.emb_layer_norm_after is not None:
            x = model.emb_layer_norm_after(x)
        
        # (T, B, E) => (B, T, E)
        x = x.transpose(0, 1)
        
        # Mean pooling
        mask = ~tokens.eq(model.padding_idx)
        mask_float = mask.unsqueeze(-1).float()
        sum_repr = (x * mask_float).sum(dim=1)
        count = mask_float.sum(dim=1).clamp(min=1e-9)
        protein_repr = sum_repr / count
        
        return protein_repr
    
    def forward(self, tokens, nlp_embedding, batch_size,
                return_attention_weights=False):
        """
        End-to-End forward pass.
        
        Args:
            tokens: [B, L] tokenized protein sequences
            nlp_embedding: [num_labels, nlp_dim] GO text embedding
            batch_size: actual batch size
            return_attention_weights: keep interface compatibility (no gating weights used)
            
        Returns:
            logits: [B, num_labels]
        """
        num_labels = nlp_embedding.size(0)
        
        # ESM2 + dual Adapter encode sequence
        seq_repr = self.forward_esm(tokens)  # [B, embed_dim]
        
        # Expand sequence representation to [B * num_labels, embed_dim]
        seq_expanded = seq_repr.unsqueeze(1).expand(-1, num_labels, -1)
        seq_flat = seq_expanded.reshape(-1, seq_expanded.size(-1))
        
        # Project
        seq_feat = self.seq_proj(seq_flat)  # [B * num_labels, hidden_dim=512]
        nlp_feat = self.nlp_proj(nlp_embedding)  # [num_labels, hidden_dim=512]
        
        # Expand NLP features
        nlp_feat_expanded = nlp_feat.unsqueeze(0).expand(batch_size, -1, -1)
        nlp_feat_batched = nlp_feat_expanded.reshape(-1, nlp_feat.size(-1))  # [B * num_labels, 512]
        
        # Concatenation fusion: H_fused = [h_seq' ∥ h_label'] ∈ R^1024
        fused_feat = torch.cat([seq_feat, nlp_feat_batched], dim=-1)  # [B * num_labels, 1024]
        
        # Classification MLP
        logits = self.classifier(fused_feat)  # [B * num_labels, 1]
        logits = logits.view(batch_size, num_labels)
        
        if return_attention_weights:
            # Maintain interface compatibility, return None as weights
            return logits, None
        return logits
    
    def print_trainable_params(self):
        """Print trainable parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print(f"EndToEndMZSGO-DA Parameter Summary:")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"\nTrainable components:")
        
        groups = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                group = name.split('.')[0]
                if group not in groups:
                    groups[group] = 0
                groups[group] += param.numel()
        
        for group, count in sorted(groups.items()):
            print(f"  {group}: {count:,}")
        print(f"{'='*60}\n")