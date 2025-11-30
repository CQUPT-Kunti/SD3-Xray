#!/bin/bash

export MODEL_NAME="../checkpoint/sd3_medium_incl_clips.safetensors"
export INSTANCE_DIR="../data"
export OUTPUT_DIR="../out1"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export CUDA_VISIBLE_DEVICES=1

accelerate launch train.py \
  --safetensors_path ../checkpoint/sd3_medium_incl_clips.safetensors \
  --sd3_config_dir ./sd3_config_cache \
  --instance_data_dir ../data \
  --output_dir ./output_full_80k \
  --resolution 1024 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --lr_scheduler cosine \
  --lr_warmup_steps 2000 \
  --max_train_steps 80000 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --save_steps 4000 \
  --validation_steps 1000





