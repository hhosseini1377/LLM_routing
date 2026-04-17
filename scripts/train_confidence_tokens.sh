#!/bin/bash
# Train confidence tokens baseline on HotpotQA.
# Single GPU: python -m baselines.confidence_tokens
# Multi-GPU DDP: ./scripts/train_confidence_tokens.sh [nproc]

set -e
cd "$(dirname "$0")/.."

source ./env/bin/activate 2>/dev/null || true

export HF_HOME=${HF_HOME:-/data/gpfs/projects/punim2662/.cache/huggingface}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/gpfs/projects/punim2662/.cache/triton}

NPROC=${1:-1}
[ -n "$1" ] && shift  # drop first arg (nproc) so remaining args pass through

if [ "$NPROC" -gt 1 ]; then
    echo "Launching DDP training with $NPROC GPUs (torchrun)"
    torchrun --nproc_per_node="$NPROC" -m baselines.confidence_tokens "$@"
else
    echo "Launching single-GPU training"
    python -m baselines.confidence_tokens "$@"
fi
