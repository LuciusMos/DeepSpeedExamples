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

deepspeed --num_gpus 1 main.py \
   --data_path phd_qualified_seeds \
   --data_split 2,4,4 \
   --model_name_or_path bigscience/bloomz-560m \
   --num_padding_at_beginning 0 \
   --gradient_accumulation_steps 4 \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log