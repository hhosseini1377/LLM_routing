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

#!/bin/bash

# --- BASELINE PARAMETERS (Fixed for all runs) ---
BASE_ARGS=" \
  --num_epochs 10 \
  --batch_size 64 \
  --data_size None \
  --dataset mmlu \
  --is_cpx_token_trainable True \
  --cpx_aggregation mean \
  --dropout_rate 0.1 \
  --classifier_dropout True \
  --use_lora True \
  --mask_lora_for_non_cpx True \
  --use_class_weights True \
  --class_weight_power 0.5 \
  --embedding_lr 5e-5 \
  --embedding_weight_decay 0.0 \
  --evaluation_size None \
  --scheduler cosine \
  --warmup_steps 0.1 \
  --gradient_checkpointing True \
  --max_grad_norm 1.0 \
  --patience 5 \
  --amsgrad False \
  --label_smoothing 0.15 \
  --lora_alpha 32 \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 20 \
  --cpx_tokens [CPX1] [CPX2] \
  --metric f1 \
"

# --- EXPERIMENT 1: Current Optimized Baseline (FFN-Only) ---
# Goal: Re-run with the stabilized schedule (10 epochs, 0.1 warmup, patience 5)
echo "--- Running Experiment 1: Optimized FFN-Only Baseline ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.15 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 2: Max Capacity (r=32, Low Reg) ---
# Goal: Increase adapter complexity to maximize F1 potential.
echo "--- Running Experiment 2: Max Capacity (r=32, Low Reg) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.001 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 3: Full Structural Targets (Attention + FFN) ---
# Goal: Test if adding Q/O projections improves feature extraction.
echo "--- Running Experiment 3: Full Targets (Attention + FFN) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.1 \
  --context_window 512 \
  --lora_target_modules q_proj o_proj gate_proj up_proj down_proj

# --- EXPERIMENT 4: Attention Bias (Q,V,K,O) ---
# Goal: Focus purely on adapting the Attention communication path, r=16.
echo "--- Running Experiment 4: Attention Bias (Q,V,K,O) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.15 \
  --context_window 512 \
  --lora_target_modules q_proj v_proj k_proj o_proj

# --- EXPERIMENT 5: Ultra-Low Regularization (FFN) ---
# Goal: Test if removing almost all constraints allows the model to learn aggressively.
echo "--- Running Experiment 5: Ultra-Low Regularization ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.0 \
  --lora_r 16 \
  --lora_dropout 0.0 \
  --label_smoothing 0.0 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 6: High Regularization / Low Capacity ---
# Goal: Test if a small adapter constrained by high dropout generalizes better on the small dataset.
echo "--- Running Experiment 6: High Reg / Low Capacity (r=8) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.05 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.25 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 7: Classifier Dominance (Aggressive Classifier LR) ---
# Goal: Test if a very fast classifier can quickly leverage the latent features of the FFN-adapted backbone.
echo "--- Running Experiment 7: Classifier Dominance ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 3e-4 \
  --aggregator_lr 1e-5 \
  --lora_lr 3e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.15 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 8: Full Context Window (1024) ---
# Goal: Ensure all long GSM8K prompts are fully processed without truncation.
echo "--- Running Experiment 8: Full Context Window (1024) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.1 \
  --context_window 1024 \
  --lora_target_modules gate_proj up_proj down_proj

# --- EXPERIMENT 9: Hybrid Targets (Q + FFN, r=24) ---
# Goal: Combine the most critical reasoning (FFN) and feature extraction (Q) layers with moderate capacity.
echo "--- Running Experiment 9: Hybrid Targets (Q + FFN) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 24 \
  --lora_alpha 48 \
  --lora_dropout 0.1 \
  --context_window 512 \
  --lora_target_modules q_proj gate_proj up_proj down_proj

# --- EXPERIMENT 10: Linear Scheduler Test (5 Epochs) ---
# Goal: Test the impact of a fast, linear LR decay instead of the cosine curve.
echo "--- Running Experiment 10: Linear Scheduler (5 Epochs) ---"
python3 -m cpx_model.main \
  $BASE_ARGS \
  --num_epochs 5 \
  --scheduler 'linear' \
  --classifier_lr 1e-4 \
  --aggregator_lr 5e-5 \
  --lora_lr 6e-5 \
  --weight_decay 0.01 \
  --lora_r 16 \
  --lora_dropout 0.15 \
  --context_window 512 \
  --lora_target_modules gate_proj up_proj down_proj