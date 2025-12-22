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

# Optuna hyperparameter tuning script
# This script uses Optuna to find optimal hyperparameters
#
# Usage:
#   ./train_cpx_optuna.sh
#
# The script will:
# - Run n_trials optimization trials
# - Save results to SQLite database (can be resumed)
# - Tune specified hyperparameters (learning rates, LoRA, dropout, optimizer)
# - Save best parameters to JSON file
#
# Tuning flags:
#   --tune_learning_rates    Tune classifier_lr, aggregator_lr, embedding_lr, lora_lr
#   --tune_lora              Tune lora_r, lora_alpha, lora_dropout
#   --tune_target_modules    Tune which modules LoRA is applied to (q_proj, o_proj, etc.)
#   --tune_dropout           Tune dropout_rate, classifier_dropout
#   --tune_optimizer         Tune weight_decay, warmup_steps, label_smoothing
#   --tune_aggregation       Tune cpx_aggregation method
#
# Storage (optional):
#   --storage                SQLite database path for persisting results
#   --load_if_exists         Resume existing study if found
#   If you remove --storage, results are in-memory (lost when script exits)

python3 -m cpx_model.optuna_tune \
    --n_trials 20 \
    --study_name "cpx_optuna_study" \
    --direction 'auto' \
    --num_epochs 5 \
    --batch_size 64 \
    --data_size '10000' \
    --dataset 'mmlu' \
    --evaluation_size '1000' \
    --context_window 1024 \
    --is_cpx_token_trainable True \
    --use_lora True \
    --mask_lora_for_non_cpx True \
    --use_class_weights True \
    --class_weight_power 1 \
    --gradient_checkpointing True \
    --max_grad_norm 1.0 \
    --patience 3 \
    --scheduler 'cosine' \
    --freeze_LoRA_layers False \
    --freeze_LoRA_start_layer_idx 0 \
    --cpx_tokens [CPX1] \
    --tune_learning_rates \
    --tune_target_modules \
    --dropout_rate 0.1 \
    --classifier_dropout True \
    --weight_decay 0.01 \
    --embedding_weight_decay 0.0 \
    --warmup_steps 0.1 \
    --amsgrad False \
    --label_smoothing 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --cpx_aggregation 'mean'
    # --lora_target_modules gate_proj up_proj down_proj \


    #   --tune_lora \
#   --tune_dropout \
#   --tune_optimizer
  # --tune_aggregation \

# Fixed hyperparameters (used when corresponding tune flag is not set)
# Uncomment and modify these if you want to fix specific hyperparameters:
# --classifier_lr 3e-4 \
# --aggregator_lr 2e-4 \
# --embedding_lr 1e-4 \
# --lora_lr 1.5e-4 \\

