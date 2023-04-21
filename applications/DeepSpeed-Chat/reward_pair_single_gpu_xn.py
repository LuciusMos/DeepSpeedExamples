# -*- coding: utf-8 -*-
# liyan26@kuaishou.com 李岩 @2023-03-21 17:03:42
# Last Change:  2023-04-20 13:24:17

import sys
import json
import math

import torch
from tqdm import tqdm
from transformers import BloomTokenizerFast
from transformers import BloomForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig

if __name__ == '__main__':
    gpu_device = int(sys.argv[1])
    model_pt, inference_file, output_file = sys.argv[2:]

    pids = []
    texts = []
    chatgpt_anses = []
    with open(inference_file, 'r') as f:
        for pid, line in enumerate(f):
            #     pid, text = line.strip().split('\t')
            text = line.strip()
            query, chatgpt_ans = text.strip().split('<ANSWER>')
            texts.append(query + '<ANSWER>')
            pids.append(pid)
            chatgpt_anses.append(chatgpt_ans)
        #     if pid % 1000 == 0:
        #         print(gpu_device, pid, query[:10], chatgpt_ans[:10])
        #     line = line.strip()
        #     items = line.split('\t')
        #     if len(items) != 3:
        #         continue
        #     photoid, indus, text = items
        #     text = '<QUESTION>' + prompt_dic['root'] % (prompt_dic[indus_dic[indus]], text)
        #     #import pdb; pdb.set_trace()
        #     pids.append(photoid)
        #     texts.append(text[:1500])

    segment_len = math.ceil(len(texts) / 8.)
    _pids = pids[gpu_device * segment_len: (gpu_device + 1) * segment_len]
    _texts = texts[gpu_device * segment_len: (gpu_device + 1) * segment_len]
    _chatgpt_anses = chatgpt_anses[gpu_device *
                                   segment_len: (gpu_device + 1) * segment_len]

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
    model_config = AutoConfig.from_pretrained('bigscience/bloomz-7b1')
    model = BloomForCausalLM(model_config)
    model_static_dict = torch.load(model_pt)
    model.load_state_dict(model_static_dict)
    model.cuda()
    model.eval()

    #     checkpoint = "/share/ad/liuguoyu/code/model_pretrain/snapshots/63d20535b4ec8c846757259ba2360f207a149210"
    #     model_static_dict = torch.load(
    #         "/share/ad/liuguoyu/code/llm-test/all_indus_finetune_model.pt")
    #     tokenizer = BloomTokenizerFast.from_pretrained(checkpoint)
    #     model_config = AutoConfig.from_pretrained(checkpoint)
    #     model = BloomForCausalLM(model_config)
    #     model.load_state_dict(model_static_dict)
    #     model.cuda()
    #     model.eval()

    wf = open(output_file.format(gpu_device), 'w')
    for pid, text, chatgpt_ans in tqdm(zip(_pids, _texts, _chatgpt_anses)):
        inputs = tokenizer(text, return_tensors='pt').input_ids
        inputs = inputs.to('cuda:0')
        try:
            with torch.no_grad():
                outputs = model.generate(inputs,
                                         max_new_tokens=2000,
                                         do_sample=True,
                                         top_k=50,
                                         top_p=0.95)
            res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]
            wf.writelines(
                '{}\t{}<ANSWER-ChatGPT>{}\n'.format(pid, res, chatgpt_ans))
            wf.flush()
        except Exception as e:
            print("===!!!Exception", e)
    wf.close()

#     indus_dic = {
#         #'\N': 'unknown',
#         '\\N': 'unknown',
#         '珠宝饰品': 'ecomm',
#         '美妆个护': 'ecomm', '乐器': 'ecomm',
#         '鞋靴': 'ecomm', '箱包': 'ecomm', '食品': 'ecomm', '服装': 'ecomm', '百货': 'ecomm',
#         '汽车': 'ecomm', '母婴用品': 'ecomm', '文具教辅': 'ecomm', '家居家装': 'ecomm', '工业用品': 'ecomm',
#         '眼镜钟表': 'ecomm', '儿童用品': 'ecomm', '网购平台': 'ecomm', '文玩/艺术收藏': 'ecomm',
#         '医疗健康': 'medical', '美容美体': 'medical', '手机/电脑/数码': 'it', '电器': 'it', '游戏': 'game',
#         '休闲娱乐活动': 'entertainment', '传媒资讯': 'medium', '商务服务': 'business', '生活服务': 'life',
#         '运动户外': 'exercise', '金融': 'finance', '房地产': 'realty', '教育': 'edu', '旅游': 'travel',
#         '招商加盟': 'investment', '通信': 'communication', '出行服务': 'traffic'
#     }

#     prompt_dic = {
#         "root": "根据输入提取以下内容：%s。输入文本为：%s.",
#         "business": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、
# 情感信息是正向还是负向",
#         "communication": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有>明星、情感信息是正向还是负向",
#         "ecomm": "商品名称、商品品牌、原材料、商品产地、价格信息、适用场景、产品卖点、用户痛点、面向人群、营销方式、售后服务",
#         "edu": "品牌、课程品质、学习项目、素养提升、年级、学科、升学阶段、求学目的、入学门槛、学历含金量、学历类型、学位类型、考试公共课、报考专业、报考院校、>留学目的地、副业类型、产品卖点、用户痛点、",
#         "entertainment": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有>明星、情感信息是正向还是负向",
#         "exercise": "商品名称、商品品牌、原材料、商品产地、价格信息、适用场景、产品卖点、用户痛点、面向人群、营销方式、售后服务",
#         "finance": "金融产品、公司性质、贷款主体、贷款额度、申请资料、保险周期、赔付额度、投资期限、收益类型、抵押方式、还款方式、软件名称、面向人群、服务卖点",
#         "game": "游戏名称、游戏题材、游戏类型、游戏玩法、画面描述、营销信息、是否氪金、体现的世界观、对抗性、是否社交属性、游戏刺激点、游戏角色名、游戏亮点、表
# 达手法、叙述方式、剧情简介、所属平台、角色IP",
#         "investment": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星
# 、情感信息是正向还是负向",
#         "it": "商品名称、商品品牌、原材料、商品产地、价格信息、适用场景、产品卖点、用户痛点、面向人群、营销方式、售后服务",
#         "life": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、情感
# 信息是正向还是负向",
#         "medical": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、>情感信息是正向还是负向",
#         "medium": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、情
# 感信息是正向还是负向",
#         "realty": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、情
# 感信息是正向还是负向",
#         "traffic": "产品名称、产品类型、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、情感信息是>正向还是负向",
#         "travel": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、情
# 感信息是正向还是负向",
#         "unknown": "产品名称、产品类型、产品功效、产品卖点、面向人群、用户痛点、人物设定、营销方式、是否有行动号召、叙事方式、开头吸引注意的方式、是否有明星、>情感信息是正向还是负向"
#     }
