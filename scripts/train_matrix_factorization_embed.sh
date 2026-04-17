#!/usr/bin/env bash
# Matrix-factorization router training with sentence-transformer embeddings.
# Usage:
#   ./scripts/train_matrix_factorization_embed.sh mmlu_original_pro_auxiliary_gsm8k
#   ./scripts/train_matrix_factorization_embed.sh lmsys_chat1m
# Extra args are forwarded to Python (e.g. --epochs 20 --embed_model all-mpnet-base-v2).

set -euo pipefail
cd "$(dirname "$0")/.."

source ./env/bin/activate 2>/dev/null || true

DATASET="${1:-hotpotqa}"
[ -n "${1:-}" ] && shift

export HF_HOME="${HF_HOME:-/data/gpfs/projects/punim2662/.cache/huggingface}"

echo "Training matrix factorization embed router: dataset=$DATASET"
python -m matrix_factorizatoin.train_hotpotqa_embed --dataset "$DATASET" "$@"
