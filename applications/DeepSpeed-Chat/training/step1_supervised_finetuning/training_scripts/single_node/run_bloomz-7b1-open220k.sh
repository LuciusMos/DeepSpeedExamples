#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path open_domain_220k \
   --data_split 10,0,0 \
   --model_name_or_path bigscience/bloomz-7b1 \
   --model_cache /data/zhaoliangxuan/model_zoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 5e-5 \
   --weight_decay 0.0 \
   --num_train_epochs 50 \
   --save_iter 1000 \
   --gradient_accumulation_steps 2 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 0 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --lora_dim 128 \
   --lora_module_name decoder.layers. \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log


# bash ./training_scripts/single_node/run_bloomz-7b1-open220k.sh 'open220k' ''