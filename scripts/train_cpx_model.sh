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

# Limit threading to prevent OOM when using more CPUs
# These variables prevent libraries from auto-detecting all CPUs and spawning too many threads
# Set to a reasonable value (e.g., 2-4 threads per process) to avoid excessive memory usage
# export OMP_NUM_THREADS=2
# export MKL_NUM_THREADS=2
# export NUMEXPR_NUM_THREADS=2
# export TORCH_NUM_THREADS=2

# --- BASELINE PARAMETERS (Fixed for all runs) ---
BASE_ARGS=" \
  --num_epochs 10 \
  --batch_size 64 \
  --data_size None \
  --is_cpx_token_trainable True \
  --cpx_aggregation attention \
  --classifier_dropout True \
  --use_lora False \
  --mask_lora_for_non_cpx True \
  --embedding_weight_decay 0.0 \
  --evaluation_size None \
  --gradient_checkpointing True \
  --patience 2 \
  --amsgrad False \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 20 \
  --cpx_tokens [CPX1] \
  --metric roc_auc \
  --use_class_weights True \
  --use_weighted_sampling False \
  --weighting_strategy label \
  --oversample_factor 1 \
  --sampling_weight_power 1.0 \
  --class_weight_power 1.0 \
  --dataset_name mmlu_original_pro_auxiliary \
  --dataset_model_name qwen8b \
  --model_name allenai/OLMoE-1B-7B-0924 \
  --save_model True \
"


# Config 1: Baseline (Full LoRA + Weighted Sampling) with 18 layers
# IMPROVED VERSION: Optimized for MNLI multi-class classification
# Changes: Increased patience, max_grad_norm; reduced LRs for full fine-tuning; increased epochs
# Note: Context window 256 is sufficient (MNLI examples avg ~41 tokens, max ~240 tokens)
echo "--- Config 1: Baseline (Full LoRA + Weighted Sampling) - IMPROVED FOR MNLI ---"
python3 -m cpx_model.main \
  ${BASE_ARGS} \
  --dropout_rate 0.1 \
  --scheduler cosine \
  --warmup_steps 0.1 \
  --max_grad_norm 0.3 \
  --label_smoothing 0.08 \
  --lora_alpha 64 \
  --classifier_lr 8e-5 \
  --aggregator_lr 7e-5 \
  --lora_lr 4e-5 \
  --embedding_lr 4e-5 \
  --weight_decay 0.002 \
  --lora_r 32 \
  --lora_dropout 0.05 \
  --context_window 768 \
  --num_labels 1 \
  --patience 5 \
  --num_epochs 10 \
  --lora_target_modules q_proj o_proj k_proj v_proj \
  --use_last_hidden_state_baseline False 
