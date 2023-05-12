#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# You can provide two models to compare the performance of the sft and the final model
export CUDA_VISIBLE_DEVICES=0
python inference_compare.py \
    --output_file goliath-full-dschat.json \
    --model_name_or_path_sft /data/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/goliath-stage1-mt/e1-i6000 \
    --language goliath-open-domain \
    --max_new_tokens 2048 \
    --test_sample_num -1

#    --model_name_or_path_sft /data/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/full-dolly-accelerator/eN-iN/ \
    # --model_name_or_path_final_ema /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/bloomz-560m-523-leftpad/step3-models/bloomz-7b1/actor_ema/ \
    # --model_name_or_path_final /home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/bloomz-560m-523-leftpad/step3-models/bloomz-7b1/actor/ \
