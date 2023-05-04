# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os
import json

from transformers import AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model  # noqa
from utils.utils import set_random_seed, load_hf_tokenizer  # noqa

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with the trained models")
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to output the inference results",
        required=True,
    )
    # parser.add_argument(
    #     "--model_name_or_path_baseline",
    #     type=str,
    #     help="Path to baseline model",
    # )
    # parser.add_argument(
    #     "--model_baseline_cache",
    #     type=str,
    #     default=None,
    #     help="Path to cached baseline model",
    # )
    # parser.add_argument(
    #     "--model_name_or_path_comparison_list",
    #     nargs='*',
    #     help="Path to the models you wish to compare with, e.g. SFT model, EMA model, etc.",
    # )
    parser.add_argument(
        "--model_name_or_path_sft",
        type=str,
        help="Path to sft model",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_final",
        type=str,
        help="Path to final model after 3 steps",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path_final_ema",
        type=str,
        default=None,
        help="Path to final EMA model after 3 steps",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',  # todo
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Chinese", "Japanese", "phd", "keyword"],
    )
    parser.add_argument(
        "--test_sample_num",
        type=int,
        default=30,
        help='How many samples will be used for testing. [-1 means ALL samples]',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="A seed for reproducible training.",
    )

    args = parser.parse_args()
    return args


def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):

    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)

    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    return result


def print_utils(gen_output):
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


def prompt_eval(args, model_sft, model_final, model_final_ema, tokenizer, device, prompts, with_gt=False):
    f = open(args.output_file, 'w')
    for p_index, prompt in enumerate(prompts):
        if with_gt:
            complete_prompt = prompt
            prompt = prompt['prompt']
        print("==========   prompt  =========")
        print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print("==========SFT: Greedy=========")
        r_base = generate(model_sft,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)[0]
        # PhD spcified
        r_base = r_base.split('<ANSWER>')[1]
        print_utils(r_base)
        print("==========final: Greedy=========")
        r_final_g = generate(model_final,
                             tokenizer,
                             inputs,
                             num_beams=1,
                             num_return_sequences=args.num_return_sequences,
                             max_new_tokens=args.max_new_tokens)[0]
        # PhD spcified
        r_final_g = r_final_g.split('<ANSWER>')[1]
        print_utils(r_final_g)
        if model_final_ema is not None:
            print("========final-EMA: Greedy========")
            r_final_ema_g = generate(model_final_ema,
                                     tokenizer,
                                     inputs,
                                     num_beams=1,
                                     num_return_sequences=args.num_return_sequences,
                                     max_new_tokens=args.max_new_tokens)[0]
            # PhD spcified
            r_final_ema_g = r_final_ema_g.split('<ANSWER>')[1]
            print_utils(r_final_ema_g)
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========final: Multinomial sampling=========")
        # r_final_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_final_m)
        # print("==========final: Beam Search=========")
        # r_final_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_final_b)
        # print("==========final: Beam-search multinomial sampling=========")
        # r_final_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_final_s)
        # print("==========final: Diverse Beam Search=========")
        # r_final_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_final_d)
        # print("==========final: Constrastive Search=========")
        # r_final_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_final_c)
        print("====================prompt end=============================")
        json_string = {
            'index': p_index,
            'prompt': prompt,
            'sft': r_base,
            'final': r_final_g,
        }
        if model_final_ema is not None:
            json_string['final_ema']: r_final_ema_g
        if with_gt:
            json_string['chatgpt'] = complete_prompt['chosen']
        f.write(json.dumps(json_string, ensure_ascii=False, indent=4) + '\n')

    f.close()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda:0")
    # step 1 & 2 use right-padding, step 3 uses left-padding
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline, fast_tokenizer=True, padding_side="left")

    # model_baseline = create_hf_model(AutoModelForCausalLM,
    #                             args.model_name_or_path_baseline,
    #                             tokenizer,
    #                             None,
    #                             model_cache=args.model_baseline_cache)
    model_sft = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path_sft,
                                tokenizer,
                                None)
    model_final = create_hf_model(AutoModelForCausalLM,
                                  args.model_name_or_path_final,
                                  tokenizer,
                                  None)
    model_final_ema = None
    if args.model_name_or_path_final_ema is not None:
        model_final_ema = create_hf_model(AutoModelForCausalLM,
                                          args.model_name_or_path_final_ema,
                                          tokenizer,
                                          None)
    model_sft.to(device)
    model_final.to(device)
    if model_final_ema is not None:
        model_final_ema.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # finald models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    if args.language == "English":
        prompts = [
            "Human: Please tell me about Microsoft in a few sentence? Assistant:",
            "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
            "Human: Write a short poem about a wise frog. Assistant:",
            "Human: Who was president of the United States in 1955? Assistant:",
            "Human: How does a telescope work? Assistant:",
            "Human: Why do birds migrate south for the winter? Assistant:"
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]
    elif args.language == "phd":
        prompts = []
        with open('/share/zhaoliangxuan/dataset/phd_qualified_seeds_clean.csv', 'r') as f:
            for i, line in enumerate(f):
                pid, text = line.strip().split('\t')
                query, answers = text.strip().split('<ANSWER>')
                ours_answer, chatgpt_answer = answers.split('<ANSWER-ChatGPT>')
                prompts.append({
                    'prompt': query + '<ANSWER>',
                    'chosen': chatgpt_answer,
                    'rejected': ours_answer,
                })
                if i == args.test_sample_num:
                    break
    elif args.language == "keyword":
        prompts = []
        with open('/share/zhaoliangxuan/dataset/keyword.json', 'r') as f:
            for i, line in enumerate(f.readlines()):
                prompts.append(json.loads(line))
                if i == args.test_sample_num:
                    break
        prompts = [p['prompt'] for p in prompts]

    prompt_eval(
        args=args,
        model_sft=model_sft,
        model_final=model_final,
        model_final_ema=model_final_ema,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        with_gt=(args.language == "phd")
    )


if __name__ == "__main__":
    main()
