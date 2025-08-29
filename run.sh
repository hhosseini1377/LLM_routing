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

export HUGGINGFACE_TOKEN="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"
# python main.py --model_name distilbert  --num_epochs 5 --batch_size 32 --context_window 512 --data_size None --strategy cls
# python3 test.py
# python3 -m router_system.run_server --model_name TheBloke/Mistral-7B-Instruct-v0.1-AWQ --utilization 0.5 --dtype 'float16'
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 8000:8000 -v $(pwd):/app -e HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN run_server
# sudo docker build -t run_server .