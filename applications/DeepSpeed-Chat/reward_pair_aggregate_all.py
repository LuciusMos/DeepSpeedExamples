import tqdm

if __name__ == '__main__':
    seperate_files = "./dataset/phd_qualified_seeds_inference_output_{}.csv"
    pids = []
    texts = []
    for i in range(8):
        with open(seperate_files.format(i), 'r') as f:
            for lineno, line in enumerate(f):
                if line.strip().split('\t')[0].isdigit():
                    pid, text = line.strip().split('\t')
                    text = text.strip()
                else:
                    text += line.strip()
                    if '<ANSWER>' in text and '<ANSWER-ChatGPT>' in text:
                        texts.append(text)
                        pids.append(pid)

    wf = open("./dataset/phd_qualified_seeds_inference_output_all.csv", 'w')
    for pid, text in tqdm(zip(pids, texts)):
        try:
            wf.writelines('{}\t{}\n'.format(pid, text))
            wf.flush()
        except Exception as e:
            print("===!!!Exception", e)
    wf.close()
