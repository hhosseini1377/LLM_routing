#!/bin/bash

source ./env/bin/activate

# Change the cache directory for huggingface
    
export TRITON_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/triton
export XDG_CONFIG_HOME=/data/gpfs/projects/punim2662/.config
export VLLM_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/vllm
export VLLM_CACHE_ROOT=/data/gpfs/projects/punim2662/.cache/vllm
export TORCH_HOME=/data/gpfs/projects/punim2662/.cache/torch/
export TORCHINDUCTOR_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/torch/inductor
export CUDA_CACHE_PATH=/data/gpfs/projects/punim2662/.cache/nvidia/
export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

export HF_AUTH_TOKEN="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"

python3 -m bert_routing.main \
    --model_name bert \
    --num_epochs 10\
    --batch_size 32 \
    --context_window 512 \
    --data_size '10000' \
    --evaluation_size '1000' \
    --strategy 'cls' \
    --dataset 'mmlu' \
    --dropout_rate 0.1 \
    --classifier_dropout False \
    --weight_decay 0.01 \
    --layers_to_freeze 6 \
    --freeze_layers False \
    --scheduler 'cosine' \
    --warmup_steps 0.1 \
    --max_grad_norm 1.0 \
    --embedding_lr 1e-5 \
    --classifier_lr 1e-5 \
    --model_lr 1e-5 \
    --freeze_embedding False \