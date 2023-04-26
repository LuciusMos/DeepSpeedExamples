#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the baseline and the finetuned model
export CUDA_VISIBLE_DEVICES=0
python prompt_eval.py \
    --model_name_or_path_baseline bigscience/bloomz-7b1 \
    --model_name_or_path_baseline /data/model_zoo \
    --model_name_or_path_finetune /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/keyword-bloomz-7b1-sft/actor-models/bloomz-7b1 \
    --language keyword \
    --max_new_tokens 200 \
    # --language phd \
    # --max_new_tokens 600 \

