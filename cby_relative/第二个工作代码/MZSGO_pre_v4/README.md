# MZSGO-DA: Domain-Aware Adapter Pretraining for Protein Function Prediction

## 方法概述

MZSGO-DA（MZSGO with Domain-Aware Adapter pretraining）通过在ESM2的Transformer层中注入轻量级适配器模块，在冻结主干参数的前提下，通过**双通道对比学习**将InterPro域文本语义和GO功能定义语义引入序列编码过程，从而增强序列表征与功能预测目标之间的关联。

## 两阶段训练流程

### 阶段1：域功能感知双通道对比学习预训练

- 在ESM2后16层注入adapter_0，冻结ESM2原始参数
- **序列-域通道**：将序列表示与InterPro域文本嵌入对齐
- **序列-功能通道**：将序列表示与GO功能文本嵌入对齐
- 总损失：`L_pretrain = λ_dom * L_CL^dom + λ_func * L_CL^func`（λ_dom=λ_func=0.5）
- 缺乏域注释的蛋白质仅参与序列-功能通道
- 三个独立投影头：ProjHead_seq, ProjHead_dom, ProjHead_func
- 投影维度 d_p = 256，中间隐藏维度 512
- 温度超参数 τ = 0.07

### 阶段2：双适配器端到端下游训练

- 冻结adapter_0，注入新的可训练adapter_1
- 两组Adapter在同一层并行运行，输出均值融合
- **拼接融合策略**：
  - 序列特征投影：h_seq ∈ R^1280 → h_seq' ∈ R^512
  - 标签文本投影：E_label ∈ R^2560 → h_label' ∈ R^512
  - 模态丢弃：仅对序列特征，p=0.15
  - 拼接：H_fused = [h_seq' ∥ h_label'] ∈ R^1024
  - 分类MLP：H_fused → LN → GELU → Dropout → 1
- 推理时仅需输入蛋白质序列和候选GO标签文本，无需显式域嵌入

## 核心设计

### 1. Adapter嵌入Transformer层内部

Adapter在Transformer层中有两个注入位置：
- 注入位置①：多头自注意力输出之后
- 注入位置②：前馈网络输出之后

```
Self-Attention → Adapter_attn → residual →
FFN → Adapter_ffn → residual
```

### 2. 适配器模块（ResMLP）

```
Adapter(x) = LN(W_up · ReLU(W_down · x)) + x
```
其中 d=1280, d_b=640

### 3. 多组Adapter融合

预训练阶段 G=1，下游阶段 G=2：
```
X_adapter = (1/G) * Σ Adapter^(g)(X)
```

### 4. 下游拼接融合（vs 第三章门控融合）

相较于MZSGO的三模态门控融合：
- 本章仅涉及序列与标签两路输入
- 序列表征已通过适配器预训练注入了域功能先验
- 信息融合的复杂度较低，采用拼接方式即可满足需求

## 文件结构

```
MZSGO_pre_v4/
├── esm_adapter.py          # 核心: ResMLP + 动态注入Adapter到ESM2
├── pretrain_model.py       # 预训练模型 (双通道对比学习: seq-dom + seq-func)
├── pretrain_dataset.py     # 预训练数据集 (支持域+功能嵌入，保留无域样本)
├── pretrain.py             # 预训练脚本 (DDP, 支持λ_dom/λ_func)
├── model_downstream.py     # 下游模型 (End-to-End 双Adapter + 拼接融合)
├── main_ddp.py             # 下游训练脚本 (DDP)
├── run_pretrain.sh         # 预训练启动脚本
├── run_downstream.sh       # 下游训练启动脚本
├── run_pipeline.sh         # 完整流水线
└── utils/                  # 工具函数
```

## 快速开始

```bash
# 完整流水线（双通道预训练 + 下游End-to-End + 零样本测试）
bash run_pipeline.sh

# 仅预训练
bash run_pretrain.sh

# 仅下游训练（需先完成预训练）
bash run_downstream.sh
```

## 参数量统计

对于ESM2-650M + 16层Adapter：

| 组件 | 参数量 | 状态 |
|------|--------|------|
| ESM2原始参数 | ~650M | 冻结 |
| adapter_0（16层×2个） | ~53M | 冻结 |
| adapter_1（16层×2个） | ~53M | 可训练 |
| 特征投影与分类头 | ~2M | 可训练 |
| 总计 | ~758M | — |
| 可训练参数 | ~55M | 7.3% |

## 显存注意

End-to-End训练时ESM2在forward中参与计算（虽然参数冻结），
显存占用比静态embedding方式大。建议：
- batch_size_train: 4 per GPU
- gradient_accumulation_steps: 4（补偿小batch）
- 有效batch = 4 × 6GPUs × 4accum = 96
