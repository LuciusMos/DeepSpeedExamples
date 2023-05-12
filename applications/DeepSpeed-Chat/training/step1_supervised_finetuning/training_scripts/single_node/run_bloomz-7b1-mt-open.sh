#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
DATASET=$3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path $DATASET \
   --data_split 10,0,0 \
   --model_name_or_path bigscience/bloomz-7b1-mt \
   --model_cache /data/zhaoliangxuan/model_zoo \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 2e-5 \
   --weight_decay 0.0 \
   --num_train_epochs 3 \
   --save_iter 1000 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 300 \
   --seed 0 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log

#   --lora_dim 128 \
#   --lora_module_name decoder.layers. \
# bash ./training_scripts/single_node/run_bloomz-7b1-mt-open.sh phoenix-690k-7b-mt-lr2e-5-batch4-step8-warm300-lora128 3 phoenix-690k
# bash ./training_scripts/single_node/run_bloomz-7b1-mt-open.sh phoenix-200k-7b-mt-lr2e-5-batch4-step8-warm300-lora128 3 phoenix-200k
# bash ./training_scripts/single_node/run_bloomz-7b1-mt-open.sh phoenix-200k-7b-mt-lr2e-5-batch4-step8-warm300-nolora 3 phoenix-200k
# bash ./training_scripts/single_node/run_bloomz-7b1-mt-open.sh 70k_preprocess-7b-mt-lr2e-5-batch4-step8-warm300-lora128 3 70k_preprocess
# bash ./training_scripts/single_node/run_bloomz-7b1-mt-open.sh phoenix-origin-7b-mt-lr2e-5-batch4-step8-warm300-nolora 3 phoenix-origin
