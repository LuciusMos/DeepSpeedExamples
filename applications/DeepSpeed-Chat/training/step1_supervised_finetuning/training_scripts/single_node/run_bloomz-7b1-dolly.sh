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
   --model_name_or_path bigscience/bloomz-7b1 \
   --model_cache /data/zhaoliangxuan/model_zoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 2048 \
   --learning_rate 5e-5 \
   --weight_decay 0.0 \
   --save_iter 200 \
   --num_train_epochs 3 \
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

# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'dolly' '' 'Goliath-Stage1-Dolly'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'dolly-lr2e-5' '' 'Goliath-Stage1-Dolly'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'cot-dolly' '' 'Goliath-Stage1-Dolly_CoT'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'chat-dolly' '' 'Goliath-Stage1-Dolly_Chat'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'selfinst-dolly' '' 'Goliath-Stage1-Dolly_Selfinst'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'full-dolly' '' 'Goliath-Stage1-Full'
# bash ./training_scripts/single_node/run_bloomz-7b1-dolly.sh 'full-dolly-accelerator' '' 'Goliath-Stage1-Full'

# export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com && cd /data/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat && pip install -r requirements.txt && pip install deepspeed