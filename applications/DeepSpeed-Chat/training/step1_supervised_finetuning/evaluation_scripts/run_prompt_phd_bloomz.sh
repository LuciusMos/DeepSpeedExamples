#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the sft and the final model
export CUDA_VISIBLE_DEVICES=0
python inference_compare.py \
    --model_name_or_path_sft /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/bloomz-560m-523-a100/actor-models/bloomz-7b1 \
    --model_name_or_path_final /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/bloomz-560m-523-a100/step3-models/bloomz-7b1/actor \
    --language phd \
    --max_new_tokens 900 \
    --test
