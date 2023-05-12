raw_datasets = []
with open('/home/zhaoliangxuan/DeepSpeedExamples/applications/DeepSpeed-Chat/dataset/phd_qualified_seeds_inference_output_all.csv', 'r') as f:
    for line in f:
        try:
            pid, text = line.strip().split('\t')
            query, answers = text.strip().split('<ANSWER>')
            ours_answer, chatgpt_answer = answers.split('<ANSWER-ChatGPT>')
            raw_datasets.append({
                'promt': query + '<ANSWER>',
                'chosen': chatgpt_answer,
                'rejected': ours_answer,
            })
        except Exception as e:
            print(e, text.count('<ANSWER>'), text, len(text.split('<ANSWER>')))
            break
