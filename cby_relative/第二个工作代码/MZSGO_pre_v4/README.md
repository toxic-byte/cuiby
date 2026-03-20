# MZSGO-DA v4: Internal Adapter + End-to-End Training

## 相比 v3 的核心改进

### 1. Adapter 嵌入 Transformer 层内部（最关键！）

**v3**: Adapter 串联在 Transformer 层外面
```
Transformer_Layer → Adapter → 下一层
（Adapter 替换了整层输出，serial）
```

**v4**: Adapter 嵌入 Transformer 层内部（attn后 + FFN后）
```
Self-Attention → Adapter_attn → residual →
FFN → Adapter_ffn → residual
（Adapter 与原始路径并行，parallel）
```

### 2. 残差连接修正

**v3**: `LayerNorm(adapter_out + residual)` → 16层累积导致分布偏移 (cos_sim=0.27)

**v4**: `module(x) + x`（LayerNorm在module内部，残差是纯净加法）→ 保持原始ESM2分布

### 3. 正常梯度反传

**v3**: `detach() + FrozenLayerForward (identity Jacobian近似)` → 梯度信号严重失真

**v4**: ESM2参数冻结(`requires_grad=False`)但保持在计算图中，梯度正常通过

### 4. End-to-End 下游训练

**v3**: 预训练 → 提取静态embedding → 下游用固定embedding训练

**v4**: 预训练 → 下游End-to-End（adapter_0冻结 + adapter_1可训练 + 分类头），每次forward都经过ESM2

### 5. 双 Adapter 策略

- **adapter_0**（预训练阶段学习的domain信息）→ 冻结
- **adapter_1**（下游任务新加的task-specific适配）→ 可训练
- 两组Adapter在同一层并行，输出取平均

## 文件结构

```
MZSGO_pre_v4/
├── esm_adapter.py          # 核心: ResMLP + 动态注入Adapter到ESM2
├── pretrain_model.py       # 预训练模型 (ESM2+Adapter + 对比学习)
├── pretrain.py             # 预训练脚本 (DDP)
├── model_downstream.py     # 下游模型 (End-to-End 双Adapter)
├── main_ddp.py             # 下游训练脚本 (DDP)
├── run_pretrain.sh         # 预训练启动脚本
├── run_downstream.sh       # 下游训练启动脚本
├── run_pipeline.sh         # 完整流水线
├── utils/ -> ../MZSGO_pre_v3/utils/   # 共享工具
├── data/ -> ../MZSGO_pre_v3/data/     # 共享数据
└── pretrain_dataset.py -> ../MZSGO_pre_v3/pretrain_dataset.py
```

## 快速开始

```bash
# 完整流水线（预训练 + 下游End-to-End + 零样本测试）
bash run_pipeline.sh

# 仅预训练
bash run_pretrain.sh

# 仅下游训练（需先完成预训练）
bash run_downstream.sh
```

## 预训练流水线简化

v3 是4个阶段：
1. 预训练 → 2. 提取embedding → 3. 下游训练 → 4. 测试

v4 简化为2个阶段：
1. 预训练 → 2. End-to-End下游训练（自动评估）

省掉了"提取embedding"这一步，因为End-to-End训练不需要静态embedding。

## 显存注意

End-to-End训练时ESM2在forward中参与计算（虽然参数冻结），
显存占用比v3的静态embedding方式大。建议：
- batch_size_train: 4 per GPU（v3用16）
- gradient_accumulation_steps: 4（补偿小batch）
- 有效batch = 4 × 7GPUs × 4accum = 112（与v3一致）

## 技术细节

### Adapter 注入机制
不需要Fork整个ESM2代码。通过 `inject_adapters_into_esm2()` 在运行时
动态替换 `TransformerLayer` 为 `AdapterTransformerLayer`。

### 参数量
对于ESM2-650M + 16层Adapter：
- ESM2原始参数: ~650M (冻结)
- adapter_0: ~32个ResMLP (每层2个: attn后+FFN后) × (1280→640→1280+LN) ≈ 约53M
- adapter_1: 同上 ≈ 约53M
- 分类头: ~2M
- **可训练**: adapter_1 + 分类头 ≈ 55M
