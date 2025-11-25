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

BASE_ARGS=" \
    --model_name bert \
    --num_epochs 10\
    --batch_size 16 \
    --context_window 512 \
    --data_size None \
    --evaluation_size None \
    --strategy cls \
    --dataset combined \
    --scheduler cosine \
    --metric roc_auc \
"

python3 -m bert_routing.main \
    $BASE_ARGS \
    --classifier_dropout True \
    --weight_decay 0.01 \
    --layers_to_freeze 16 \
    --freeze_layers False \
    --warmup_steps 0.1 \
    --max_grad_norm 1.0 \
    --embedding_lr 3e-5 \
    --classifier_lr 3e-5 \
    --model_lr 3e-5 \
    --freeze_embedding False \
    --dropout_rate 0.2 \
    --use_weighted_sampling True \
    --dataset_weight_power 1.0 \
    --class_weight_power 1.0 \

    python3 -m bert_routing.main \
    $BASE_ARGS \
    --classifier_dropout True \
    --weight_decay 0.01 \
    --layers_to_freeze 16 \
    --freeze_layers False \
    --warmup_steps 0.1 \
    --max_grad_norm 1.0 \
    --embedding_lr 3e-5 \
    --classifier_lr 3e-5 \
    --model_lr 3e-5 \
    --freeze_embedding False \
    --dropout_rate 0.2 \    

    python3 -m bert_routing.main \
    $BASE_ARGS \
    --classifier_dropout True \
    --weight_decay 0.01 \
    --layers_to_freeze 16 \
    --freeze_layers False \
    --warmup_steps 0.1 \
    --max_grad_norm 1.0 \
    --embedding_lr 3e-5 \
    --classifier_lr 3e-5 \
    --model_lr 3e-5 \
    --freeze_embedding False \
    --dropout_rate 0.2 \