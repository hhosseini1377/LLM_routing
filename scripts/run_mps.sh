#!/bin/bash

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$(id -un) 
nvidia-cuda-mps-control -d

# BASE_ARGS=" \
#     --model_name deberta \
#     --num_epochs 5\
#     --batch_size 64 \
#     --context_window 256 \
#     --data_size None \
#     --evaluation_size None \
#     --strategy cls \
#     --scheduler cosine \
#     --metric f1 \
# "

# python3 -m bert_routing.main \
#     $BASE_ARGS \
#     --classifier_dropout True \
#     --weight_decay 0.01 \
#     --layers_to_freeze 16 \
#     --freeze_layers False \
#     --warmup_steps 0.1 \
#     --max_grad_norm 1.0 \
#     --embedding_lr 3e-5 \
#     --classifier_lr 3e-5 \
#     --model_lr 3e-5 \
#     --freeze_embedding False \
#     --dropout_rate 0.1 \
#     --use_class_weights True \
#     --use_weighted_sampling False \
#     --dataset_weight_power 1.0 \
#     --sampling_weight_power 1.0 \
#     --loss_weight_power 1.0 \
#     --class_weight_power 1.0 \
#     --dataset_name anli \
#     --dataset_model_name qwen8b \