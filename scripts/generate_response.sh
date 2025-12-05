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


# vllm serve Qwen/Qwen3-8B \
#   --max-model-len 4096 \
#   --max-num-batched-tokens 65536 \
#   --tensor-parallel-size 2 \
#   --host 0.0.0.0 \
#   --port 8000

python3 -m routing_dataset.run_prompts

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