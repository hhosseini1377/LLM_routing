#!/bin/bash
# Run HotpotQA success prediction pipeline (generate + self-evaluate with Qwen3-8B)

set -e
cd "$(dirname "$0")/.."

source ./env/bin/activate 2>/dev/null || true

# Cache directories
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/gpfs/projects/punim2662/.cache/triton}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-/data/gpfs/projects/punim2662/.config}
export VLLM_CACHE_DIR=${VLLM_CACHE_DIR:-/data/gpfs/projects/punim2662/.cache/vllm}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/data/gpfs/projects/punim2662/.cache/vllm}
export TORCH_HOME=${TORCH_HOME:-/data/gpfs/projects/punim2662/.cache/torch}
export HF_HOME=${HF_HOME:-/data/gpfs/projects/punim2662/.cache/huggingface}

python -m routing_dataset.run_hotpotqa_success_prediction_vllm \
    --input_file routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_val.pkl \
    --output_file routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_val_success_pred.pkl \
    --model Qwen/Qwen3-8B \
    --max_tokens_answer 256 \
    --max_tokens_eval 16 \
    --tensor_parallel_size 1 \
    "$@"
