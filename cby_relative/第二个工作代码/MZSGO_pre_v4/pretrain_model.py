"""
预训练模型（v4版本 - MZSGO-DA）

双通道对比学习预训练：
1. 序列-域通道：将InterPro域文本语义注入adapter
2. 序列-功能通道：将GO功能定义语义注入adapter
3. 总损失 = λ_dom * L_CL^dom + λ_func * L_CL^func

Adapter 嵌入 ESM2 Transformer 层内部（attn后 + FFN后），正常梯度反传。
ESM2 参数冻结但保持在计算图中（梯度正常流过，只是不更新）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

from esm_adapter import (
    inject_adapters_into_esm2,
    freeze_esm_parameters,
    get_adapter_state_dict,
    load_adapter_state_dict,
)


class ProjectionHead(nn.Module):
    """
    投影头：将编码器输出映射到对比学习的共享空间。
    使用2层MLP + LayerNorm + GELU + Dropout。
    
    结构：Linear → LN → GELU → Dropout → Linear → LN
    中间隐藏维度为512，输出向量经L2归一化后用于计算余弦相似度。
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ESM2WithAdapter(nn.Module):
    """
    ESM2 + 内嵌 Adapter 模型。
    
    Adapter 嵌入在 Transformer 层内部（self-attention 后 + FFN 后），
    通过正常的 PyTorch autograd 进行梯度反传。
    ESM2 的原始参数冻结（requires_grad=False），但不需要 detach 或 identity 近似。
    """
    def __init__(self, esm_type='esm2_t33_650M_UR50D',
                 num_adapter_layers=16,
                 bottleneck_dim=None,
                 adapter_dropout=0.0,
                 adapter_names=None):
        super().__init__()
        
        self.esm_type = esm_type
        self.num_adapter_layers = num_adapter_layers
        
        # 加载原始 ESM2 模型
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
        
        if adapter_names is None:
            adapter_names = ['adapter_0']
        
        # ★ 核心改动：将 Adapter 注入 ESM2 的 Transformer 层内部
        self.adapter_params = inject_adapters_into_esm2(
            self.esm_model,
            num_adapter_layers=num_adapter_layers,
            adapter_names=adapter_names,
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            adapter_dropout=adapter_dropout,
        )
        
        # ★ 冻结 ESM2 原始参数，只训练 Adapter
        freeze_esm_parameters(self.esm_model, self.adapter_params)
        
        self.batch_converter = self.alphabet.get_batch_converter()
    
    def forward(self, tokens, return_residue_repr=False):
        """
        前向传播。
        
        Args:
            tokens: [B, L] tokenized 序列
            return_residue_repr: 是否返回残基级别表示
            
        Returns:
            protein_repr: [B, embed_dim] 蛋白质级别表示 (mean pooling)
            residue_repr (optional): [B, L, embed_dim] 残基级别表示
        """
        assert tokens.ndim == 2
        
        # 直接调用 ESM2 的 forward，获取最后一层的表示
        results = self.esm_model(tokens, repr_layers=[self.num_layers], 
                                  return_contacts=False)
        
        # 获取最后一层表示: [B, L, embed_dim]
        residue_repr = results["representations"][self.num_layers]
        
        # Mean pooling（排除 padding 和特殊 token）
        padding_mask = tokens.eq(self.esm_model.padding_idx)  # [B, L]
        mask = ~padding_mask
        mask_float = mask.unsqueeze(-1).float()  # [B, L, 1]
        sum_repr = (residue_repr * mask_float).sum(dim=1)  # [B, embed_dim]
        count = mask_float.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        protein_repr = sum_repr / count  # [B, embed_dim]
        
        if return_residue_repr:
            return protein_repr, residue_repr
        return protein_repr
    
    def get_trainable_params(self):
        """获取所有可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def get_trainable_param_names(self):
        """获取可训练参数的名称"""
        return [n for n, p in self.named_parameters() if p.requires_grad]


class DomainAwarePretrainModel(nn.Module):
    """
    Domain-Aware 预训练模型（MZSGO-DA）。
    
    通过双通道对比学习将域功能语义和GO标签文本语义共同注入序列编码器的适配器参数中。
    
    架构：
    - 序列编码器：ESM2 + 内嵌 Adapter -> ProjHead_seq -> 序列投影 z_seq
    - 域编码器：预计算的 domain embedding -> ProjHead_dom -> 域投影 z_dom
    - 功能编码器：预计算的 GO func embedding -> ProjHead_func -> 功能投影 z_func
    - 双通道对比损失：L = λ_dom * L_CL^dom + λ_func * L_CL^func
    """
    def __init__(self,
                 esm_type='esm2_t33_650M_UR50D',
                 domain_dim=2560,
                 func_dim=2560,
                 projection_hidden_dim=512,
                 projection_output_dim=256,
                 num_adapter_layers=16,
                 bottleneck_dim=None,
                 adapter_dropout=0.0,
                 temperature=0.07,
                 lambda_dom=0.5,
                 lambda_func=0.5):
        super().__init__()
        
        self.temperature = temperature
        self.lambda_dom = lambda_dom
        self.lambda_func = lambda_func
        
        # 序列编码器：ESM2 + 内嵌 Adapter
        self.seq_encoder = ESM2WithAdapter(
            esm_type=esm_type,
            num_adapter_layers=num_adapter_layers,
            bottleneck_dim=bottleneck_dim,
            adapter_dropout=adapter_dropout,
            adapter_names=['adapter_0'],  # 预训练只用一组 adapter
        )
        
        self.seq_embed_dim = self.seq_encoder.embed_dim
        
        # 序列投影头 ProjHead_seq
        self.seq_projection = ProjectionHead(
            input_dim=self.seq_embed_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            dropout=adapter_dropout
        )
        
        # 域投影头 ProjHead_dom
        self.domain_projection = ProjectionHead(
            input_dim=domain_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            dropout=adapter_dropout
        )
        
        # ★ 功能投影头 ProjHead_func（新增：用于序列-功能对比通道）
        self.func_projection = ProjectionHead(
            input_dim=func_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim,
            dropout=adapter_dropout
        )
    
    def forward(self, tokens, domain_embeddings, func_embeddings, has_domain_mask=None):
        """
        双通道对比学习前向传播。
        
        Args:
            tokens: [B, L] tokenized 蛋白质序列
            domain_embeddings: [B, domain_dim] 预计算的 domain embedding
            func_embeddings: [B, func_dim] 预计算的 GO 功能 embedding
            has_domain_mask: [B] bool tensor，标记哪些样本有域注释
                            None 时默认所有样本都有域注释
            
        Returns:
            loss: 双通道对比学习总损失
            loss_dom: 序列-域通道损失（仅有域注释的样本）
            loss_func: 序列-功能通道损失（所有样本）
            seq_proj: [B, proj_dim] 序列投影
            domain_proj: [B, proj_dim] 域投影
            func_proj: [B, proj_dim] 功能投影
        """
        # 序列编码
        seq_repr = self.seq_encoder(tokens)  # [B, embed_dim]
        
        # 投影到共享空间
        seq_proj = self.seq_projection(seq_repr)  # [B, proj_dim]
        domain_proj = self.domain_projection(domain_embeddings)  # [B, proj_dim]
        func_proj = self.func_projection(func_embeddings)  # [B, proj_dim]
        
        # L2 归一化
        seq_proj = F.normalize(seq_proj, dim=-1)
        domain_proj = F.normalize(domain_proj, dim=-1)
        func_proj = F.normalize(func_proj, dim=-1)
        
        # ★ 双通道 InfoNCE 对比损失
        # 通道1：序列-域（仅有域注释的样本参与）
        if has_domain_mask is not None and has_domain_mask.sum() > 1:
            seq_proj_dom = seq_proj[has_domain_mask]
            domain_proj_dom = domain_proj[has_domain_mask]
            loss_dom = self.info_nce_loss(seq_proj_dom, domain_proj_dom)
        elif has_domain_mask is not None and has_domain_mask.sum() <= 1:
            # 不足2个有域注释的样本，无法构成对比学习
            loss_dom = torch.tensor(0.0, device=seq_proj.device)
        else:
            # has_domain_mask=None，所有样本都参与
            loss_dom = self.info_nce_loss(seq_proj, domain_proj)
        
        # 通道2：序列-功能（所有样本参与）
        loss_func = self.info_nce_loss(seq_proj, func_proj)
        
        # 总损失：加权和
        loss = self.lambda_dom * loss_dom + self.lambda_func * loss_func
        
        return loss, loss_dom, loss_func, seq_proj, domain_proj, func_proj
    
    def info_nce_loss(self, seq_proj, target_proj):
        """
        InfoNCE 对比学习损失（双向）。
        
        正样本对：同一蛋白质的 (序列投影, 目标投影)
        负样本对：不同蛋白质的 (序列投影, 目标投影)
        """
        batch_size = seq_proj.shape[0]
        logits = torch.matmul(seq_proj, target_proj.T) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_seq2target = F.cross_entropy(logits, labels)
        loss_target2seq = F.cross_entropy(logits.T, labels)
        
        return (loss_seq2target + loss_target2seq) / 2.0
    
    def extract_sequence_embedding(self, tokens):
        """预训练后，仅用序列编码器提取 domain-aware 序列表示"""
        with torch.no_grad():
            seq_repr = self.seq_encoder(tokens)
        return seq_repr
    
    def get_adapter_state_dict(self):
        """获取 Adapter 的 state_dict，用于保存和下游任务加载"""
        return get_adapter_state_dict(self.seq_encoder.adapter_params)
    
    def print_trainable_params(self):
        """打印可训练参数信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n{'='*60}")
        print(f"Model Parameter Summary (MZSGO-DA - Dual-Channel Pretrain):")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"\nTrainable parameter details:")
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape} ({param.numel():,})")
        print(f"{'='*60}\n")
