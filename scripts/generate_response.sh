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
# The script will check for HF_AUTH_TOKEN, HUGGINGFACE_TOKEN, or HF_TOKEN
if [ -z "$HF_AUTH_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ] && [ -z "$HF_TOKEN" ]; then
    echo "Warning: No Hugging Face token found in environment variables."
    echo "Please set one of: HF_AUTH_TOKEN, HUGGINGFACE_TOKEN, or HF_TOKEN"
fi


# vllm serve Qwen/Qwen3-8B \
#   --max-model-len 4096 \
#   --max-num-batched-tokens 65536 \
#   --tensor-parallel-size 2 \
#   --host 0.0.0.0 \
#   --port 8000

python3 -m routing_dataset.llm_judge

# vllm bench serve \
#   --model Qwen/Qwen3-8B \
#   --backend openai \
#   --base-url http://localhost:8000 \
#   --dataset-name custom \
#   --dataset-path ./routing_dataset/datasets/benchmarking_prompts.jsonl \
#   --num-prompts 500 \
#   --max-concurrency 64 \
#   --request-rate inf \
#   --result-filename ./qwen8b_benchmark_results.json