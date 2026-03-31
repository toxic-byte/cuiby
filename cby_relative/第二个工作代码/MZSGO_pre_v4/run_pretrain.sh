#!/bin/bash
# ============================================================
# MZSGO-DA 双通道对比学习预训练启动脚本
# 使用 DDP 在多 GPU 上进行对比学习预训练
# ============================================================

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
NPROC=6

cd "$(dirname "$0")"

LOG_DIR="./logs/pretrain"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pretrain_da_${TIMESTAMP}.log"

# 预训练参数
BATCH_SIZE=8            # 每张GPU的batch size
GRAD_ACCUM=2            # 梯度累积步数，有效batch = 8*6*2 = 96
EPOCHS=50
LR=1e-4
TEMPERATURE=0.07
NUM_ADAPTER_LAYERS=16
PROJ_DIM=256
ADAPTER_DROPOUT=0.0
SAVE_DIR="./ckpt/pretrain"

# ★ 双通道损失权重
LAMBDA_DOM=0.5
LAMBDA_FUNC=0.5

# 早停参数
PATIENCE=5
MIN_DELTA=0.0001

echo "============================================"
echo "MZSGO-DA Dual-Channel Pretraining"
echo "============================================"
echo "Key features:"
echo "  - Dual-channel: seq-domain + seq-function"
echo "  - λ_dom=${LAMBDA_DOM}, λ_func=${LAMBDA_FUNC}"
echo "  - Adapter inside Transformer (attn后 + FFN后)"
echo "  - Clean residual: module(x) + x"
echo "  - Normal gradient backprop"
echo ""
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC} GPUs)"
echo "Log: ${LOG_FILE}"
echo "Start: $(date)"
echo "============================================"

nohup torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=29500 \
    pretrain.py \
    --run_mode full \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --temperature ${TEMPERATURE} \
    --num_adapter_layers ${NUM_ADAPTER_LAYERS} \
    --projection_output_dim ${PROJ_DIM} \
    --adapter_dropout ${ADAPTER_DROPOUT} \
    --lambda_dom ${LAMBDA_DOM} \
    --lambda_func ${LAMBDA_FUNC} \
    --save_dir ${SAVE_DIR} \
    --log_dir ${LOG_DIR} \
    --save_every 5 \
    --fp16 \
    --num_workers 4 \
    --patience ${PATIENCE} \
    --min_delta ${MIN_DELTA} \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "PID: ${PID}" >> "${LOG_FILE}.pid"
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Stop:     kill ${PID}"
