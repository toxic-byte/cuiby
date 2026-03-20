#!/bin/bash
# ============================================================
# v4 下游 End-to-End 训练启动脚本
# ============================================================

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
NPROC=6

cd "$(dirname "$0")"

LOG_DIR="./logs/downstream"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/downstream_v4_${TIMESTAMP}.log"

# 下游训练参数
BATCH_SIZE=4            # ESM2在forward中，显存需求大，batch size要小
GRAD_ACCUM=4            # 梯度累积补偿小batch，有效batch = 4*7*4 = 112
EPOCHS=30
LR=1e-4
PRETRAIN_CKPT="./ckpt/pretrain/pretrain_best.pt"
PATIENCE=5              # 早停：连续5个epoch loss不降则停止
MIN_DELTA=0.0001        # 最小改善阈值

echo "============================================"
echo "v4 End-to-End Downstream Training"
echo "============================================"
echo "Key improvements:"
echo "  - End-to-End (no static embeddings)"
echo "  - Dual Adapter: adapter_0(frozen) + adapter_1(trainable)"
echo ""
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC} GPUs)"
echo "Pretrain ckpt: ${PRETRAIN_CKPT}"
echo "Log: ${LOG_FILE}"
echo "Start: $(date)"
echo "============================================"

nohup torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=29501 \
    main_ddp.py \
    --run_mode full \
    --onto all \
    --epoch_num ${EPOCHS} \
    --batch_size_train ${BATCH_SIZE} \
    --batch_size_test ${BATCH_SIZE} \
    --learning_rate ${LR} \
    --pretrain_ckpt ${PRETRAIN_CKPT} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --fp16 \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "PID: ${PID}" >> "${LOG_FILE}.pid"
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Stop:     kill ${PID}"
