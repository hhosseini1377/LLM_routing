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

  python3 -m cpx_model.main \
  --num_epochs 10 \
  --batch_size 64 \
  --data_size 'None' \
  --dataset 'mmlu' \
  --is_cpx_token_trainable True \
  --cpx_aggregation 'mean' \
  --dropout_rate 0.1 \
  --classifier_dropout True \
  --use_lora True \
  --mask_lora_for_non_cpx True \
  --use_class_weights True \
  --class_weight_power 1 \
  --classifier_lr 1.5e-4 \
  --aggregator_lr 5e-5 \
  --embedding_lr 5e-5 \
  --lora_lr 8e-5 \
  --weight_decay 0.01 \
  --embedding_weight_decay 0.0 \
  --evaluation_size 'None' \
  --context_window 512 \
  --scheduler 'cosine' \
  --warmup_steps 0.03 \
  --gradient_checkpointing True \
  --max_grad_norm 1.0 \
  --patience 3 \
  --amsgrad False \
  --label_smoothing 0.15 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.15 \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 0 \
  --cpx_tokens [CPX1] [CPX2] \
  --lora_target_modules q_proj o_proj gate_proj up_proj down_proj

  python3 -m cpx_model.main \
  --num_epochs 10 \
  --batch_size 64 \
  --data_size 'None' \
  --dataset 'mmlu' \
  --is_cpx_token_trainable True \
  --cpx_aggregation 'mean' \
  --dropout_rate 0.1 \
  --classifier_dropout True \
  --use_lora True \
  --mask_lora_for_non_cpx True \
  --use_class_weights True \
  --class_weight_power 1 \
  --classifier_lr 6e-5 \
  --aggregator_lr 5e-5 \
  --embedding_lr 6.924956718077479e-05 \
  --lora_lr 8e-5 \
  --weight_decay 0.01 \
  --embedding_weight_decay 0.0 \
  --evaluation_size 'None' \
  --context_window 512 \
  --scheduler 'cosine' \
  --warmup_steps 0.03 \
  --gradient_checkpointing True \
  --max_grad_norm 1.0 \
  --patience 3 \
  --amsgrad False \
  --label_smoothing 0.15 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.15 \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 0 \
  --cpx_tokens [CPX1] \
  --lora_target_modules q_proj o_proj gate_proj up_proj down_proj