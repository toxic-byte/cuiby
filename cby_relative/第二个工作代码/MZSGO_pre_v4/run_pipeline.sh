#!/bin/bash
# ============================================================
# MZSGO-DA 完整流水线：双通道预训练 -> End-to-End 下游训练 -> 零样本测试
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
TORCHRUN=/d/cuiby/miniconda3/envs/dplm/bin/torchrun
export http_proxy="http://127.0.0.1:7894"
export https_proxy="http://127.0.0.1:7894"

cd "$(dirname "$0")"

LOG_DIR="./logs/pipeline"
mkdir -p ${LOG_DIR}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_da_${TIMESTAMP}.log"

run_pipeline() {
    set -e

    echo "============================================================"
    echo "MZSGO-DA Full Pipeline"
    echo "============================================================"
    echo "Key features:"
    echo "  1. Dual-channel pretrain: seq-domain + seq-function"
    echo "  2. Adapter inside Transformer (attn后 + FFN后)"
    echo "  3. Clean residual: module(x) + x"
    echo "  4. End-to-End downstream (no static embeddings)"
    echo "  5. Dual Adapter: adapter_0(frozen) + adapter_1(trainable)"
    echo "  6. Concatenation fusion (seq || label → MLP)"
    echo "============================================================"
    echo "Start: $(date)"
    echo ""

    # ============================================================
    # Stage 1: 双通道对比学习预训练
    # ============================================================
    echo "============================================================"
    echo "Stage 1: Dual-Channel Contrastive Pretraining"
    echo "Start: $(date)"
    echo "============================================================"

    ${TORCHRUN} \
        --nproc_per_node=${NPROC} \
        --master_port=29500 \
        pretrain.py \
        --run_mode full \
        --batch_size 8 \
        --gradient_accumulation_steps 2 \
        --epochs 50 \
        --lr 1e-4 \
        --temperature 0.07 \
        --num_adapter_layers 16 \
        --projection_output_dim 256 \
        --adapter_dropout 0.0 \
        --lambda_dom 0.5 \
        --lambda_func 0.5 \
        --save_dir ./ckpt/pretrain \
        --log_dir ./logs/pretrain \
        --save_every 5 \
        --fp16 \
        --num_workers 4 \
        --patience 5 \
        --min_delta 0.0001

    echo "Stage 1 completed: $(date)"

    # ============================================================
    # Stage 2: End-to-End 下游训练 (full mode)
    # ============================================================
    echo ""
    echo "============================================================"
    echo "Stage 2: End-to-End Downstream Training (full mode)"
    echo "Start: $(date)"
    echo "============================================================"

    ${TORCHRUN} \
        --nproc_per_node=${NPROC} \
        --master_port=29501 \
        main_ddp.py \
        --run_mode full \
        --onto all \
        --epoch_num 30 \
        --batch_size_train 4 \
        --batch_size_test 4 \
        --learning_rate 1e-4 \
        --pretrain_ckpt ./ckpt/pretrain/pretrain_best.pt \
        --gradient_accumulation_steps 4 \
        --fp16 \
        --patience 5 \
        --min_delta 0.0001

    echo "Stage 2 completed: $(date)"

    # ============================================================
    # Stage 3: 零样本测试
    # ============================================================
    echo ""
    echo "============================================================"
    echo "Stage 3: Zero-shot Testing"
    echo "Start: $(date)"
    echo "============================================================"

    ${TORCHRUN} \
        --nproc_per_node=${NPROC} \
        --master_port=29502 \
        main_ddp.py \
        --run_mode zero \
        --onto all \
        --epoch_num 30 \
        --batch_size_train 4 \
        --batch_size_test 4 \
        --learning_rate 1e-4 \
        --pretrain_ckpt ./ckpt/pretrain/pretrain_best.pt \
        --gradient_accumulation_steps 4 \
        --fp16 \
        --patience 5 \
        --min_delta 0.0001

    echo ""
    echo "============================================================"
    echo "Full Pipeline Completed!"
    echo "End: $(date)"
    echo "============================================================"
}

echo "Launching MZSGO-DA pipeline in background..."
echo "Log: ${LOG_FILE}"

nohup bash -c "$(declare -f run_pipeline); cd $(pwd); export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; export NPROC=${NPROC}; export TORCHRUN=${TORCHRUN}; export http_proxy=${http_proxy}; export https_proxy=${https_proxy}; run_pipeline" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "PID: ${PID}" >> "${LOG_FILE}.pid"
echo "Monitor:  tail -f ${LOG_FILE}"
echo "Stop:     kill ${PID}"
