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
  --evaluation_size 'None' \
  --use_lora True \
  --classifier_lr 3e-5 \
  --embedding_lr 1e-5 \
  --lora_lr 1.5e-5 \
  --scheduler 'cosine' \
  --weight_decay 0.01 \
  --dropout_rate 0.1 \
  --classifier_dropout True \
  --mask_lora_for_non_cpx True \
  --max_grad_norm 1.0 \
  --dataset 'mmlu' \
  --freeze_LoRA_layers False \
  --freeze_LoRA_start_layer_idx 10 \
  --use_class_weights True \
  --amsgrad True

  # python3 -m cpx_model.main \
  # --num_epochs 10 \
  # --batch_size 64 \
  # --data_size 'None' \
  # --evaluation_size 'None' \
  # --use_lora True \
  # --classifier_lr 6e-5 \
  # --embedding_lr 2e-5 \
  # --lora_lr 2.5e-5 \
  # --scheduler 'cosine' \
  # --weight_decay 0.01 \
  # --dropout_rate 0.1 \
  # --classifier_dropout True \
  # --mask_lora_for_non_cpx True \
  # --max_grad_norm 1.0 \
  # --dataset 'mix' \
  # --freeze_LoRA_layers False \
  # --freeze_LoRA_start_layer_idx 10 \
  # --use_class_weights True
  
  # python3 -m cpx_model.main \
  # --num_epochs 10 \
  # --batch_size 32 \
  # --data_size 'None' \
  # --evaluation_size 'None' \
  # --use_lora True \
  # --classifier_lr 6e-5 \
  # --embedding_lr 2e-5 \
  # --lora_lr 2.5e-5 \
  # --scheduler 'cosine' \
  # --weight_decay 0.01 \
  # --dropout_rate 0.1 \
  # --classifier_dropout True \
  # --mask_lora_for_non_cpx True \
  # --max_grad_norm 1.0 \
  # --dataset 'mix' \
  # --freeze_LoRA_layers True \
  # --freeze_LoRA_start_layer_idx 16 \
  # --use_class_weights True

  # python3 -m cpx_model.main \
  # --num_epochs 10 \
  # --batch_size 32 \
  # --data_size 'None' \
  # --evaluation_size 'None' \
  # --use_lora True \
  # --classifier_lr 6e-5 \
  # --embedding_lr 2e-5 \
  # --lora_lr 2.5e-5 \
  # --scheduler 'cosine' \
  # --weight_decay 0.01 \
  # --dropout_rate 0.1 \
  # --classifier_dropout True \
  # --mask_lora_for_non_cpx True \
  # --max_grad_norm 1.0 \
  # --dataset 'mix' \
  # --freeze_LoRA_layers True \
  # --freeze_LoRA_start_layer_idx 16\
  # --use_class_weights True 