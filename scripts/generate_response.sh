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

# CUDA_VISIBLE_DEVICES=0 python3 -m generate_dataset.generate_responses
# python3 -m generate_dataset.evaluate_MMLU --config all --splits test validation dev --num_runs 5
# python3 -m generate_dataset.evaluate_mmlu_qwen --splits test --model_name Qwen/Qwen3-32B-AWQ

python3 -m test
# python3 -m generate_dataset.evaluate_mmlu_max_qwen \
#     --splits test \
#     --model_name Qwen/Qwen3-32B-AWQ \
#     --temperature 0.7 \
#     --top_p 0.9 \
#     --max_tokens 1024 \
#     --gpu_memory_utilization 0.8 \
#     --output_file ./generate_dataset/datasets/mmlu_max/mmlu_max_test_qwen_results.pkl