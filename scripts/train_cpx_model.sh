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

# --- BASELINE PARAMETERS (Fixed for all runs) ---
BASE_ARGS=" \
  --num_epochs 10 \
  --batch_size 64 \
  --data_size None \
  --is_cpx_token_trainable True \
  --cpx_aggregation attention \
  --dropout_rate 0.1 \
  --classifier_dropout True \
  --use_lora True \
  --mask_lora_for_non_cpx True \
  --embedding_weight_decay 0.0 \
  --evaluation_size None \
  --scheduler cosine \
  --warmup_steps 0.1 \
  --gradient_checkpointing True \
  --max_grad_norm 1.0 \
  --patience 3 \
  --amsgrad False \
  --label_smoothing 0.08 \
  --lora_alpha 64 \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 12 \
  --cpx_tokens [CPX1] [CPX2] [CPX3] \
  --metric f1 \
"

# --- RUN: Optimized Low-Reg (Slower Learning, Extended Time) ---
# Goal: Use lower LRs over 20 epochs (with 1024 context) to find a better, generalized minimum.
echo "--- Running Optimized Experiment: Slower Convergence ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 9e-5 \
  --lora_lr 6e-5 \
  --embedding_lr 6e-5 \
  --weight_decay 0.001 \
  --lora_r 32 \
  --lora_dropout 0 \
  --label_smoothing 0.08 \
  --context_window 768 \
  --lora_target_modules q_proj o_proj gate_proj up_proj down_proj \
  --use_class_weights True \
  --use_weighted_sampling False \
  --weighting_strategy label \
  --oversample_factor 1 \
  --class_weight_power 1.0 \
  --sampling_weight_power 1 \
  --loss_weight_power 1.0 \
  --dataset_name mmlu_original_pro_auxiliary \
  --dataset_model_name qwen17b \
  --model_name Qwen/Qwen3-1.7B \
