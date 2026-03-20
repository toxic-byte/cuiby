"""
下游任务模型（v4版本）—— End-to-End 双 Adapter 策略

核心改进（相比 pre_v3）：
1. 不再提取静态 embedding，而是 end-to-end 训练
2. 双 Adapter 策略：
   - adapter_0（预训练得来）→ 冻结，保持 domain 信息
   - adapter_1（下游新加的）→ 可训练，学习 task-specific 适配
3. ESM2 + adapter_0 + adapter_1 全部在 forward 中参与计算
4. 分类头随机初始化

这样做的优点：
- adapter_1 可以修正预训练偏差
- end-to-end 训练让模型可以根据下游任务微调表示
- 比静态 embedding 更灵活
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


class GatedFusionModule2(nn.Module):
    """
    双模态自适应门控融合模块。
    融合 domain-aware 序列特征和 GO 文本特征。
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, seq_feat, go_feat):
        concat_feat = torch.cat([seq_feat, go_feat], dim=-1)
        gate_logits = self.gate_network(concat_feat)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        stacked_feats = torch.stack([seq_feat, go_feat], dim=1)
        batch_size, num_feats, hidden_dim = stacked_feats.shape
        
        stacked_feats_flat = stacked_feats.view(-1, hidden_dim)
        transformed_feats_flat = self.feature_transform(stacked_feats_flat)
        transformed_feats = transformed_feats_flat.view(batch_size, num_feats, hidden_dim)
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)
        fused_feat = (transformed_feats * gate_weights_expanded).sum(dim=1)
        
        return fused_feat, gate_weights


class FeatureDropout2(nn.Module):
    """双模态 Feature Dropout"""
    def __init__(self, dropout_prob=0.15):
        super().__init__()
        self.dropout_prob = dropout_prob
    
    def forward(self, seq_feat, go_feat):
        if not self.training:
            return seq_feat, go_feat
        
        batch_size = seq_feat.size(0)
        device = seq_feat.device
        mask = (torch.rand(batch_size, 1, device=device) > self.dropout_prob).float()
        seq_feat = seq_feat * mask
        
        return seq_feat, go_feat


class EndToEndMZSGO(nn.Module):
    """
    End-to-End MZSGO 下游任务模型（v4版本）。
    
    ★ 核心改进：不再使用静态 embedding，而是 end-to-end 训练。
    
    架构：
    - ESM2 + adapter_0(冻结) + adapter_1(可训练) → 序列表示
    - LLM 编码的 GO 文本 embedding → 文本表示
    - Gated Fusion → 分类
    
    双 Adapter 策略：
    - adapter_0：预训练阶段学习的 domain 信息 → 冻结
    - adapter_1：下游任务新加的 task-specific 适配 → 可训练
    """
    def __init__(self, esm_type='esm2_t33_650M_UR50D',
                 nlp_dim=2560, hidden_dim=512, dropout=0.3,
                 num_adapter_layers=16, bottleneck_dim=None,
                 adapter_dropout=0.0,
                 pretrain_adapter_ckpt=None,
                 feature_dropout_prob=0.15):
        super().__init__()
        
        # === ESM2 + 双 Adapter ===
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
        
        # ★ 注入双 Adapter：adapter_0(预训练) + adapter_1(下游)
        self.adapter_params = inject_adapters_into_esm2(
            self.esm_model,
            num_adapter_layers=[num_adapter_layers, num_adapter_layers],
            adapter_names=['adapter_0', 'adapter_1'],
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_dropout=adapter_dropout,
        )
        
        # 冻结 ESM2 原始参数
        freeze_esm_parameters(self.esm_model, self.adapter_params)
        
        # ★ 加载预训练的 adapter_0 并冻结
        if pretrain_adapter_ckpt is not None:
            self._load_pretrain_adapter(pretrain_adapter_ckpt)
        
        # ★ 冻结 adapter_0，只训练 adapter_1
        freeze_adapter_group(self.adapter_params, 'adapter_0', freeze=True)
        freeze_adapter_group(self.adapter_params, 'adapter_1', freeze=False)
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # === 下游分类头 ===
        self.seq_proj = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.nlp_proj = nn.Sequential(
            nn.Linear(nlp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.feature_dropout = FeatureDropout2(dropout_prob=feature_dropout_prob)
        
        self.gated_fusion = GatedFusionModule2(
            hidden_dim=hidden_dim,
            dropout=dropout * 0.7
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, 1)
    
    def _load_pretrain_adapter(self, ckpt_path):
        """
        加载预训练的 adapter_0 参数。
        
        预训练 checkpoint 的 adapter 参数名形如：
            layer_17.adapter_0.0.module.0.weight  (attention 后的 ResMLP)
            layer_17.adapter_0.1.module.0.weight  (FFN 后的 ResMLP)
        
        下游模型也有相同的命名（因为 adapter_names=['adapter_0', 'adapter_1']），
        所以直接 strict=False 加载即可（adapter_1 的参数不在 checkpoint 中，保持随机初始化）。
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
            # 从完整 checkpoint 中提取 adapter 参数
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
            # 预期 adapter_1 的参数会 missing（因为它们是新加的）
            adapter_1_missing = [k for k in missing if 'adapter_1' in k]
            other_missing = [k for k in missing if 'adapter_1' not in k]
            print(f"  adapter_1 keys (expected missing): {len(adapter_1_missing)}")
            if other_missing:
                print(f"  WARNING: Other missing keys: {other_missing}")
    
    def forward_esm(self, tokens):
        """
        通过 ESM2 + 双 Adapter 获取序列表示。
        
        这里不用 results["representations"]，而是手动实现 forward
        以更好地控制内存（跳过 LM Head 等不需要的部分）。
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
        
        # 遍历所有层（已经被替换为 AdapterTransformerLayer）
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
        End-to-End 前向传播。
        
        Args:
            tokens: [B, L] tokenized 蛋白质序列
            nlp_embedding: [num_labels, nlp_dim] GO 文本 embedding
            batch_size: 实际 batch 大小
            return_attention_weights: 是否返回门控权重
            
        Returns:
            logits: [B, num_labels]
        """
        num_labels = nlp_embedding.size(0)
        
        # ★ ESM2 + 双 Adapter 编码序列
        seq_repr = self.forward_esm(tokens)  # [B, embed_dim]
        
        # 扩展序列表示到 [B * num_labels, embed_dim]
        seq_expanded = seq_repr.unsqueeze(1).expand(-1, num_labels, -1)
        seq_flat = seq_expanded.reshape(-1, seq_expanded.size(-1))
        
        # 投影
        seq_feat = self.seq_proj(seq_flat)
        nlp_feat = self.nlp_proj(nlp_embedding)
        
        # 扩展 NLP 特征
        nlp_feat_expanded = nlp_feat.unsqueeze(0).expand(batch_size, -1, -1)
        nlp_feat_batched = nlp_feat_expanded.reshape(-1, nlp_feat.size(-1))
        
        # Feature Dropout
        seq_feat, nlp_feat_batched = self.feature_dropout(seq_feat, nlp_feat_batched)
        
        # 门控融合
        fused_feat, gate_weights = self.gated_fusion(seq_feat, nlp_feat_batched)
        
        # MLP + 分类
        fused = self.fusion(fused_feat)
        logits = self.classifier(fused)
        logits = logits.view(batch_size, num_labels)
        
        if return_attention_weights:
            return logits, gate_weights
        return logits
    
    def print_trainable_params(self):
        """打印可训练参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print(f"EndToEndMZSGO Parameter Summary (v4):")
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
