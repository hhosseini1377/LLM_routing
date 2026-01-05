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

# Hugging Face authentication token - set via environment variable before running this script
# Example: export HF_AUTH_TOKEN="your_token_here" before running
if [ -z "$HF_AUTH_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "Warning: No Hugging Face token found in environment variables."
    echo "Please set one of: HF_AUTH_TOKEN, HUGGINGFACE_TOKEN, or HF_TOKEN"
fi

BASE_ARGS=" \
    --model_name deberta \
    --num_epochs 5\
    --batch_size 64 \
    --context_window 256 \
    --data_size None \
    --evaluation_size None \
    --strategy cls \
    --scheduler cosine \
    --metric f1 \
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
    --dropout_rate 0.1 \
    --use_class_weights True \
    --use_weighted_sampling False \
    --dataset_weight_power 1.0 \
    --sampling_weight_power 1.0 \
    --loss_weight_power 1.0 \
    --class_weight_power 1.0 \
    --dataset_name anli \
    --dataset_model_name qwen8b \