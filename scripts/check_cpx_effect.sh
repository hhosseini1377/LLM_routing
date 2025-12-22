#!/bin/bash

# Script to get hidden states from normal Mistral 7B model (without CPX tokens)
# 
# Usage examples:
#   Basic usage:
#     ./scripts/check_cpx_effect.sh
#
#   With custom dataset/model:
#     ./scripts/check_cpx_effect.sh \
#       --dataset_path ./path/to/dataset.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3
#
#   With custom parameters:
#     ./scripts/check_cpx_effect.sh \
#       --dataset_path ./path/to/dataset.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --max_length 1024 \
#       --num_samples 10 \
#       --output_path ./custom_output.pkl

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

# Run normal model hidden states generation
python3 -m cpx_model.check_cpx_effect \
    --dataset_path ./generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --max_length 1024

