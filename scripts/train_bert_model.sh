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

CUDA_VISIBLE_DEVICES=0 python3 -m bert_routing.main  --model_name deberta  --num_epochs 10 --batch_size 32 --context_window 512 --data_size None --strategy cls --dataset mix
