#!/usr/bin/env python
# encoding: utf-8
# Last Change:  2023-04-18 16:59:58
import sys
import torch
from transformers import BloomForCausalLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from colossalai.utils import save_checkpoint, load_checkpoint

valid_file, out_file, model_file = sys.argv[1:]

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
model_config = AutoConfig.from_pretrained('bigscience/bloomz-7b1')
model = BloomForCausalLM(model_config)
model_static_dict = torch.load(model_file)
model.load_state_dict(model_static_dict)
model.cuda()
model.eval()

texts = []
pids = []
with open(valid_file, 'r') as f:
    for line in f:
        pid, text = line.strip().split('\t')
        query, _ = text.strip().split('<ANSWER>')
        texts.append(query + '<ANSWER>')
        pids.append(pid)

outfile = open(out_file, 'w')
for pid, text in zip(pids, texts):
    inputs = tokenizer(text, return_tensors="pt").input_ids
    inputs = inputs.to('cuda:0')
    # outputs = model.generate(inputs, max_new_tokens=500, do_sample=False, num_beams=1)
    outputs = model.generate(inputs, max_new_tokens=1000, do_sample=True, top_k=50, top_p=0.95)
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)[-1]
    out = res
    print(pid, out)
    outfile.writelines('%s\t%s\n' % (pid, out))
    outfile.flush()
outfile.close()
