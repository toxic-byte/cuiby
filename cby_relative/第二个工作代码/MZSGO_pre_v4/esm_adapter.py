"""
ESM2 Adapter 模块（MZSGO-DA）

核心设计：
1. Adapter 嵌入 Transformer 层内部（attn后 + FFN后）
2. 残差连接：Adapter(x) = LN(W_up · ReLU(W_down · x)) + x
3. 正常梯度反传，ESM2参数冻结但保持在计算图中
4. 支持多组 Adapter 并行（用于下游双 Adapter 策略，输出均值融合）

每个Transformer层包含两个适配器实例，一组适配器共包含 2*N_layer 个ResMLP模块。
默认仅对ESM2最后 N_adapter=16 层注入适配器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import esm
from collections import defaultdict


# ============================================================
# ResMLP: Adapter 的基础构建块
# ============================================================

class ResMLP(nn.Module):
    """
    残差 MLP Adapter，借鉴 S-PLM 的设计。
    
    结构：Linear → ReLU → Linear → LayerNorm + 残差
    关键：LayerNorm 在 module 输出上，残差直接加 input
          即 return LayerNorm(Linear(ReLU(Linear(x)))) + x
    
    这与 pre_v3 的 LayerNorm(out + residual) 有本质区别：
    - v3: LayerNorm(adapter_out + input) → 重新归一化整个向量，16层累积导致分布偏移
    - v4: adapter_module(x) + input → 只做简单加法，保持原始分布
    """
    def __init__(self, embed_dim, bottleneck_dim=None, dropout=0.0):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = embed_dim // 2
        
        self.module = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x):
        out = self.module(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out + x  # ★ 纯净残差：adapter_output + input


# ============================================================
# AdapterTransformerLayer: 替换 ESM2 的 TransformerLayer
# ============================================================

class AdapterTransformerLayer(nn.Module):
    """
    带 Adapter 的 Transformer 层，完全替换 ESM2 的原始 TransformerLayer。
    
    与原版区别：在 self-attention 后和 FFN 后各插入 Adapter。
    多个 Adapter（如 adapter_0, adapter_1）的输出取平均后加到残差路径。
    
    这样 Adapter 就嵌入了 Transformer 内部，参与正常的梯度反传。
    """
    def __init__(self, original_layer, adapter_dict=None):
        """
        Args:
            original_layer: ESM2 的原始 TransformerLayer
            adapter_dict: nn.ModuleDict, key=adapter名, value=nn.ModuleList([ResMLP_attn, ResMLP_ffn])
                          如果为 None，行为与原始层完全相同
        """
        super().__init__()
        # 保留原始层的所有子模块
        self.self_attn = original_layer.self_attn
        self.self_attn_layer_norm = original_layer.self_attn_layer_norm
        self.fc1 = original_layer.fc1
        self.fc2 = original_layer.fc2
        self.final_layer_norm = original_layer.final_layer_norm
        
        # Adapter
        self.adapter_dict = adapter_dict
    
    def forward(self, x, self_attn_mask=None, self_attn_padding_mask=None, 
                need_head_weights=False):
        """
        完全遵循 ESM2 TransformerLayer 的 Pre-LN 结构，
        在 attention 输出和 FFN 输出后插入 Adapter。
        """
        # === Self-Attention Block ===
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        
        # ★ Adapter 插入点1：Self-Attention 输出之后，残差连接之前
        if self.adapter_dict is not None and len(self.adapter_dict) > 0:
            adapter_outputs = []
            for adapter_modules in self.adapter_dict.values():
                # adapter_modules[0] 是 attention 后的 ResMLP
                out = adapter_modules[0](x) / len(self.adapter_dict)
                adapter_outputs.append(out)
            x = torch.sum(torch.stack(adapter_outputs), dim=0)
        
        x = residual + x
        
        # === FFN Block ===
        residual = x
        x = self.final_layer_norm(x)
        x = _gelu(self.fc1(x))
        x = self.fc2(x)
        
        # ★ Adapter 插入点2：FFN 输出之后，残差连接之前
        if self.adapter_dict is not None and len(self.adapter_dict) > 0:
            adapter_outputs = []
            for adapter_modules in self.adapter_dict.values():
                # adapter_modules[1] 是 FFN 后的 ResMLP
                out = adapter_modules[1](x) / len(self.adapter_dict)
                adapter_outputs.append(out)
            x = torch.sum(torch.stack(adapter_outputs), dim=0)
        
        x = residual + x
        
        return x, attn


def _gelu(x):
    """ESM2 使用的 GELU 实现"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# ============================================================
# 核心函数：动态给 ESM2 注入 Adapter
# ============================================================

def inject_adapters_into_esm2(esm_model, num_adapter_layers=16, 
                               adapter_names=None, embed_dim=1280,
                               bottleneck_dim=None, adapter_dropout=0.0):
    """
    动态将 Adapter 注入到 ESM2 模型的 Transformer 层中。
    
    不需要 fork ESM2 代码，而是在运行时替换 TransformerLayer。
    
    Args:
        esm_model: 原始 ESM2 模型 (esm.pretrained.esm2_t33_650M_UR50D() 返回的)
        num_adapter_layers: 从最后一层开始，有多少层加 Adapter
                           可以是 int（所有 adapter 组共享）或 list（每组独立指定）
        adapter_names: Adapter 的名称列表，如 ['adapter_0'] 或 ['adapter_0', 'adapter_1']
                       默认为 ['adapter_0']
        embed_dim: ESM2 的 embedding 维度
        bottleneck_dim: Adapter 的 bottleneck 维度
        adapter_dropout: Adapter 的 dropout
    
    Returns:
        adapter_params: nn.ModuleDict，包含所有新创建的 Adapter 参数
                        结构: {f"layer_{i}": nn.ModuleDict({adapter_name: nn.ModuleList([ResMLP, ResMLP])})}
    """
    if adapter_names is None:
        adapter_names = ['adapter_0']
    
    if bottleneck_dim is None:
        bottleneck_dim = embed_dim // 2
    
    num_layers = len(esm_model.layers)
    
    # 处理 num_adapter_layers: int → 所有 adapter 共享; list → 每组独立
    if isinstance(num_adapter_layers, int):
        num_adapter_layers_per_group = [num_adapter_layers] * len(adapter_names)
    else:
        assert len(num_adapter_layers) == len(adapter_names)
        num_adapter_layers_per_group = num_adapter_layers
    
    # 构建 layer → adapter 映射
    layer_adapter_map = defaultdict(list)
    for adapter_idx, (name, n_layers) in enumerate(
            zip(adapter_names, num_adapter_layers_per_group)):
        start_layer = max(0, num_layers - n_layers)
        for layer_idx in range(start_layer, num_layers):
            layer_adapter_map[layer_idx].append(name)
    
    # 创建 Adapter 参数并替换 TransformerLayer
    adapter_params = nn.ModuleDict()
    
    for layer_idx in range(num_layers):
        original_layer = esm_model.layers[layer_idx]
        
        if layer_idx in layer_adapter_map:
            # 创建该层的 Adapter
            adapter_dict = nn.ModuleDict()
            for adapter_name in layer_adapter_map[layer_idx]:
                adapter_dict[adapter_name] = nn.ModuleList([
                    ResMLP(embed_dim, bottleneck_dim, adapter_dropout),  # attention 后
                    ResMLP(embed_dim, bottleneck_dim, adapter_dropout),  # FFN 后
                ])
            
            # 替换为带 Adapter 的层
            esm_model.layers[layer_idx] = AdapterTransformerLayer(
                original_layer, adapter_dict
            )
            
            # 记录 Adapter 参数（用于 checkpoint 保存/加载）
            adapter_params[f"layer_{layer_idx}"] = adapter_dict
        else:
            # 无 Adapter 的层也替换（保持接口一致，但 adapter_dict=None）
            esm_model.layers[layer_idx] = AdapterTransformerLayer(
                original_layer, adapter_dict=None
            )
    
    return adapter_params


def freeze_esm_parameters(esm_model, adapter_params):
    """
    冻结 ESM2 的所有原始参数，只保留 Adapter 参数可训练。
    
    正常梯度反传：冻结参数不产生梯度更新，但不影响 Adapter 参数的反传。
    不需要 FrozenLayerForward 或 detach！
    """
    # 先冻结所有参数
    for param in esm_model.parameters():
        param.requires_grad = False
    
    # 解冻 Adapter 参数
    for param in adapter_params.parameters():
        param.requires_grad = True


def freeze_adapter_group(adapter_params, adapter_name, freeze=True):
    """
    冻结或解冻某一组 Adapter。
    用于下游任务：冻结预训练 adapter_0，只训练新的 adapter_1。
    
    Args:
        adapter_params: inject_adapters_into_esm2 返回的 nn.ModuleDict
        adapter_name: 要操作的 Adapter 名称，如 'adapter_0'
        freeze: True=冻结, False=解冻
    """
    for layer_key, layer_adapters in adapter_params.items():
        if adapter_name in layer_adapters:
            for param in layer_adapters[adapter_name].parameters():
                param.requires_grad = not freeze


def get_adapter_state_dict(adapter_params, adapter_name=None):
    """
    提取特定 Adapter 组的 state_dict。
    
    Args:
        adapter_params: nn.ModuleDict
        adapter_name: 如果指定，只提取该组的参数；如果为 None，提取全部
    
    Returns:
        dict: adapter 参数的 state_dict
    """
    if adapter_name is None:
        return adapter_params.state_dict()
    
    result = {}
    for key, value in adapter_params.state_dict().items():
        if adapter_name in key:
            result[key] = value
    return result


def load_adapter_state_dict(adapter_params, state_dict, adapter_name_mapping=None, 
                             strict=False):
    """
    加载 Adapter 参数，支持名称映射。
    
    用于下游任务加载预训练 adapter：
    - 预训练时只有 adapter_0
    - 下游时有 adapter_0 + adapter_1
    - adapter_0 从 checkpoint 加载，adapter_1 随机初始化
    
    Args:
        adapter_params: nn.ModuleDict
        state_dict: 要加载的参数字典
        adapter_name_mapping: 名称映射字典，如 {'adapter_0': 'adapter_0'}
        strict: 是否严格匹配
    """
    if adapter_name_mapping is not None:
        new_state_dict = {}
        for key, value in state_dict.items():
            for src_name, dst_name in adapter_name_mapping.items():
                if src_name in key:
                    new_key = key.replace(src_name, dst_name)
                    new_state_dict[new_key] = value
                    break
        state_dict = new_state_dict
    
    missing, unexpected = adapter_params.load_state_dict(state_dict, strict=strict)
    return missing, unexpected
