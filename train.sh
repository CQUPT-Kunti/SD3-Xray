#!/bin/bash

export MODEL_NAME="../checkpoint/sd3_medium_incl_clips.safetensors"
export INSTANCE_DIR="../data"
export OUTPUT_DIR="../out1"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=0

# 先跑这个，确认修复有效
accelerate launch train.py \
    --safetensors_path ../checkpoint/sd3_medium_incl_clips.safetensors \
    --sd3_config_dir ./sd3_config_cache \
    --instance_data_dir ../data \
    --output_dir ./output_debug \
    --resolution 1024 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 500 \
    --max_train_steps 30000 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --save_steps 3000 \
    --validation_steps 500 \
    --use_8bit_adam \

