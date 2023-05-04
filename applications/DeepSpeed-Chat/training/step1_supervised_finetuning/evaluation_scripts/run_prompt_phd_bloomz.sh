#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the sft and the final model
export CUDA_VISIBLE_DEVICES=0
python inference_compare.py \
    --output_file old-leftpad-phd-compare.json \
    --model_name_or_path_sft /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/old-exps/bloomz-560m-523-leftpad/actor-models/bloomz-7b1/ \
    --model_name_or_path_final /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/old-exps/bloomz-560m-523-leftpad/step3-models/bloomz-7b1/actor/ \
    --model_name_or_path_final_ema /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/old-exps/bloomz-560m-523-leftpad/step3-models/bloomz-7b1/actor_ema/ \
    --language phd \
    --max_new_tokens 900 \
    --test_sample_num 30
