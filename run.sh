#!/bin/bash

source ./env/bin/activate

# Change the cache directory for huggingface
    
# export TRITON_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/triton
# export XDG_CONFIG_HOME=/data/gpfs/projects/punim2662/.config
# export VLLM_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/vllm
# export VLLM_CACHE_ROOT=/data/gpfs/projects/punim2662/.cache/vllm
# export TORCH_HOME=/data/gpfs/projects/punim2662/.cache/torch/
# export TORCHINDUCTOR_CACHE_DIR=/data/gpfs/projects/punim2662/.cache/torch/inductor
# export CUDA_CACHE_PATH=/data/gpfs/projects/punim2662/.cache/nvidia/
# export HF_HOME=/data/gpfs/projects/punim2662/.cache/huggingface

export HF_AUTH_TOKEN="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"


python3 -m bert_routing.main  --model_name bert  --num_epochs 10 --batch_size 16 --context_window 512 --data_size None --strategy cls
# python3 test.py
# python3 -m cpx_model.cpxmistral.train_mistral
# python3 -m cpx_model.cpxmistral.main  --num_epochs 5 --batch_size 16 --context_window 8192 --data_size 'None' --evaluation_size 'None'
# uvicorn router_system.main:app --reload 
