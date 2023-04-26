#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
python rw_eval.py \
    --model_name_or_path XXXXXX \
    --num_padding_at_beginning 0
