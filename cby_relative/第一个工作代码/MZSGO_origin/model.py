import torch
import torch.nn as nn
import math
from tqdm import tqdm
import torch.nn.functional as F


class GatedFusionModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super(GatedFusionModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, domain_feat, esm_feat, go_feat):

        concat_feat = torch.cat([domain_feat, esm_feat, go_feat], dim=-1)  # [B, hidden_dim*3]
        
        gate_logits = self.gate_network(concat_feat)  # [B, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, 3]
        
        stacked_feats = torch.stack([domain_feat, esm_feat, go_feat], dim=1)  # [B, 3, hidden_dim]
        batch_size, num_feats, hidden_dim = stacked_feats.shape
        
        stacked_feats_flat = stacked_feats.view(-1, hidden_dim)  # [B*3, hidden_dim]
        transformed_feats_flat = self.feature_transform(stacked_feats_flat)  # [B*3, hidden_dim]
        transformed_feats = transformed_feats_flat.view(batch_size, num_feats, hidden_dim)  # [B, 3, hidden_dim]
        
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [B, 3, 1]
        fused_feat = (transformed_feats * gate_weights_expanded).sum(dim=1)  # [B, hidden_dim]
        
        return fused_feat, gate_weights


class FeatureDropout(nn.Module):
    def __init__(self, dropout_prob=0.15):
        super(FeatureDropout, self).__init__()
        self.dropout_prob = dropout_prob
        
    def forward(self, domain_feat, esm_feat, go_feat):
        if not self.training:
            return domain_feat, esm_feat, go_feat
        
        batch_size = domain_feat.size(0)
        device = domain_feat.device
        
        rand_vals = torch.rand(batch_size, 2, device=device)  
        
        protein_masks = (rand_vals > self.dropout_prob).float()  # [B, 2]
        
        row_sums = protein_masks.sum(dim=1)  # [B]
        zero_rows = (row_sums == 0)  # [B] bool tensor
        
        if zero_rows.any():
            num_zero_rows = zero_rows.sum().item()
            random_positions = torch.randint(0, 2, (num_zero_rows,), device=device)
            
            zero_row_indices = torch.where(zero_rows)[0]
            protein_masks[zero_row_indices, random_positions] = 1.0
        
        domain_mask = protein_masks[:, 0:1]  # [B, 1]
        esm_mask = protein_masks[:, 1:2]     # [B, 1]
        
        domain_feat = domain_feat * domain_mask
        esm_feat = esm_feat * esm_mask
        
        return domain_feat, esm_feat, go_feat


class CustomModel(nn.Module):
    def __init__(self, esm_dim, nlp_dim, domain_size, hidden_dim=512, dropout=0.3,
                 feature_dropout_prob=0.15):
        super(CustomModel, self).__init__()
        
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
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
        
        self.domain_proj = nn.Sequential(
            nn.Linear(domain_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.feature_dropout = FeatureDropout(dropout_prob=feature_dropout_prob)
        
        self.gated_fusion = GatedFusionModule(
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
    
    def forward(self, esm_embedding, domain_embedding, nlp_embedding, batch_size, 
                return_attention_weights=False):
        num_labels = nlp_embedding.size(0)
        
        esm_feat = self.esm_proj(esm_embedding)  # [batch_size * num_labels, hidden_dim]
        domain_feat = self.domain_proj(domain_embedding)  # [batch_size * num_labels, hidden_dim]
        nlp_feat = self.nlp_proj(nlp_embedding)  # [num_labels, hidden_dim]
        
        nlp_feat_expanded = nlp_feat.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_labels, hidden_dim]
        nlp_feat_batched = nlp_feat_expanded.reshape(-1, nlp_feat.size(-1))  # [batch_size * num_labels, hidden_dim]
        
        domain_feat, esm_feat, nlp_feat_batched = self.feature_dropout(
            domain_feat, esm_feat, nlp_feat_batched
        )
        
        fused_feat, gate_weights = self.gated_fusion(
            domain_feat, esm_feat, nlp_feat_batched
        )  # fused_feat: [batch_size * num_labels, hidden_dim]
           # gate_weights: [batch_size * num_labels, 3]
        
        fused = self.fusion(fused_feat)  # [batch_size * num_labels, hidden_dim//2]
        
        logits = self.classifier(fused)  # [batch_size * num_labels, 1]
        
        logits = logits.view(batch_size, num_labels)
        
        if return_attention_weights:
            return logits, gate_weights
        
        return logits