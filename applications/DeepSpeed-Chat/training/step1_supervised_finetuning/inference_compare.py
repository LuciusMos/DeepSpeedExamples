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
        default=None,
        help="Path to sft model",
    )
    parser.add_argument(
        "--model_name_or_path_final",
        type=str,
        default=None,
        help="Path to final model after 3 steps",
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
        choices=["goliath-open-domain","English", "Chinese", "Japanese", "phd", "keyword"],
    )
    parser.add_argument(
        "--test_sample_num",
        type=int,
        default=-1,
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

    if model is None:
        return [None]

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

    if model is None:
        return [None]

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
    if isinstance(gen_output, list):
        for i in range(len(gen_output)):
            print()
            print(gen_output[i])
            print()
    else:
        print()
        print(gen_output)
        print()


def prompt_eval(args, model_sft, model_final, model_final_ema, tokenizer, device, prompts, with_gt=False):

    def phd_specified_post_process(infer_string):
        if infer_string is None:
            return None
        else:
            return infer_string.split('<ANSWER>')[1]

    def general_post_process(infer_string):
        if infer_string is None:
            return None
        else:
            return infer_string.split('<A>')[1]

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
                          #   num_beams=1,  # Greedy
                          num_beams=args.num_beams, do_sample=True,  # Beam-search multinomial sampling
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)[0]
        r_base = general_post_process(r_base)
        print_utils(r_base)
        print("==========final: Greedy=========")
        r_final_g = generate(model_final,
                             tokenizer,
                             inputs,
                             #   num_beams=1,  # Greedy
                             num_beams=args.num_beams, do_sample=True,  # Beam-search multinomial sampling
                             num_return_sequences=args.num_return_sequences,
                             max_new_tokens=args.max_new_tokens)[0]
        r_final_g = general_post_process(r_final_g)
        print_utils(r_final_g)
        print("========final-EMA: Greedy========")
        r_final_ema_g = generate(model_final_ema,
                                 tokenizer,
                                 inputs,
                                 #   num_beams=1,  # Greedy
                                 num_beams=args.num_beams, do_sample=True,  # Beam-search multinomial sampling
                                 num_return_sequences=args.num_return_sequences,
                                 max_new_tokens=args.max_new_tokens)[0]
        r_final_ema_g = general_post_process(r_final_ema_g)
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
            'final_ema': r_final_ema_g,
        }
        if with_gt:
            json_string['chatgpt'] = complete_prompt['chosen']
        f.write(json.dumps(json_string, ensure_ascii=False, indent=4) + '\n')

    f.close()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda:0")
    # step 1 & 2 use right-padding, step 3 uses left-padding
    model_save_path = args.model_name_or_path_sft
    if model_save_path is None:
        model_save_path = args.model_name_or_path_final_ema
    
    tokenizer = load_hf_tokenizer(model_save_path, fast_tokenizer=True, padding_side="left")

    # model_baseline = create_hf_model(AutoModelForCausalLM,
    #                             args.model_name_or_path_baseline,
    #                             tokenizer,
    #                             None,
    #                             model_cache=args.model_baseline_cache)
    model_sft = None
    model_final = None
    model_final_ema = None

    if args.model_name_or_path_sft is not None:
        model_sft = create_hf_model(AutoModelForCausalLM,
                                    args.model_name_or_path_sft,
                                    tokenizer,
                                    None)
        model_sft.to(device)
    if args.model_name_or_path_final is not None:
        model_final = create_hf_model(AutoModelForCausalLM,
                                      args.model_name_or_path_final,
                                      tokenizer,
                                      None)
        model_final.to(device)
    if args.model_name_or_path_final_ema is not None:
        model_final_ema = create_hf_model(AutoModelForCausalLM,
                                          args.model_name_or_path_final_ema,
                                          tokenizer,
                                          None)
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
    elif args.language == "goliath-open-domain":
        prompts = [
            "介绍一下快手这家公司",
            "今天周几，你是怎么判断的。",
            "根据下面的文本分析一下流浪的含义：《流浪地球》是一部2019年上映的中国科幻电影，由郭帆执导，基于刘慈欣的同名小说改编。这部电影讲述了在未来几十年内，太阳即将成为红巨星，威胁到地球的生命。为了拯救地球，人类决定采取大胆的计划，将地球推出太阳系，进入另一个恒星系寻找新家园。电影中展现了人类面临的困境和挑战，以及科技与人性的冲突与融合。这部电影在中国内地和海外市场都取得了巨大的成功，被誉为中国科幻电影的里程碑之作。",  # noqa
            "为一段探索爵士乐的历史和文化意义的YouTube视频编写脚本。",
            "这周只工作了两天, 没有什么进展, 给我写一份工作周报, 要体现我充实的工作.",
            "撰写一篇有趣的旅行博客文章，介绍最近去夏威夷的旅行经历，重点突出文化体验和必看景点。",
            "写一篇交响乐音乐会评论，讨论乐团的表现和整体观众体验。",
            "使用适当的格式来构建一封正式的推荐信，为一名申请计算机科学研究生项目的学生提供推荐。",
            "起草一封引人入胜的产品发布公告电子邮件，通知我们的客户我们的新软件解决方案。",
            "起草一封致歉邮件，向一位经历了订单延迟的客户道歉，并保证问题已得到解决。",
            "对以下信息整理成为一段流畅的一段短视频拍摄脚本内容：产品名称韩伦美额头贴、多个美女、衰老变丑痛点、不安全痛点、性价比高、用料好、视频开头体现卖点、有数字定量描述、额外赠品/福利、包退包赔、包邮包送、好评度高、使用排比手法、对比反差，行动号召购买、拍摄体现使用教程，短视频时长在60s-90s",  # noqa
            "据输入提取关键词:苹果配件全家桶，让你体验磁吸无线充的快乐！关键才200就可拿下！# 苹果配件#手机配件#数码产品 华强北六件套全新升级，理想好货 买到就是赚到！ 苹果六件套全新升级，理想好货，买到就是赚到！#苹果配件/数码配件 新升级六件套，这一套你想要的它都有，只要200轻松拿下#苹果配件/数码产品 好货买到就是赚到！ 苹果配件全新升级，超值好货，买到就是赚到！ 理想好货，买到就是赚到！关键词:",  # noqa
            "苹果配件全家桶，让你体验磁吸无线充的快乐！关键才200就可拿下！# 苹果配件#手机配件#数码产品 华强北六件套全新升级，理想好货 买到就是赚到！ 苹果六件套全新升级，理想好货，买到就是赚到！#苹果配件/数码配件 新升级六件套，这一套你想要的它都有，只要200轻松拿下#苹果配件/数码产品 好货买到就是赚到！ 苹果配件全新升级，超值好货，买到就是赚到！ 理想好货，买到就是赚到！提取上文关键词：",  # noqa
        ]
        prompts = ["<Q>" + p + "<A>" for p in prompts]
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
        with_gt=args.language == "phd",
    )


if __name__ == "__main__":
    main()
