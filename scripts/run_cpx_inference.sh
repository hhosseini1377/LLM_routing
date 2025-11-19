#!/bin/bash

# Script to run CPX model inference
# 
# Usage examples:
#   Basic usage:
#     ./scripts/run_cpx_inference.sh \
#       --model_path ./cpx_model/finetuned_models/model_mistralai_Mistral-7B-Instruct-v0.3_cpx_20251114-114450.pth \
#       --dataset_path ./generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_test.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3
#
#   With LoRA model:
#     ./scripts/run_cpx_inference.sh \
#       --model_path ./cpx_model/finetuned_models/model_mistralai_Mistral-7B-Instruct-v0.3_cpx_20251114-114450.pth \
#       --dataset_path ./generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_test.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --use_lora True \
#       --lora_r 16 \
#       --lora_alpha 32 \
#       --lora_target_modules gate_proj up_proj down_proj
#
#   With custom CPX tokens:
#     ./scripts/run_cpx_inference.sh \
#       --model_path ./cpx_model/finetuned_models/model_mistralai_Mistral-7B-Instruct-v0.3_cpx_20251114-114450.pth \
#       --dataset_path ./generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_test.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --cpx_tokens [CPX1] [CPX2]
#
#   With custom output path and batch size:
#     ./scripts/run_cpx_inference.sh \
#       --model_path ./cpx_model/finetuned_models/model_mistralai_Mistral-7B-Instruct-v0.3_cpx_20251114-114450.pth \
#       --dataset_path ./generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_test.pkl \
#       --model_name mistralai/Mistral-7B-Instruct-v0.3 \
#       --output_path ./results/inference_results.pkl \
#       --batch_size 64 \
#       --context_window 1024

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

# Run inference
python3 -m cpx_model.inference \
    --model_path ./cpx_model/finetuned_models/model_mistralai_Mistral-7B-Instruct-v0.3_cpx_20251109-214236.pth \
    --dataset_path ./generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl\
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj o_proj \
    --cpx_tokens [CPX1] [CPX2] \
    --cpx_aggregation mean \
    --output_path ./cpx_model/inference_logs
