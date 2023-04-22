import tqdm

if __name__ == '__main__':
    seperate_files = "./dataset/phd_qualified_seeds_inference_output_{}.csv"
    pids = []
    texts = []
    for i in range(8):
        with open(seperate_files.format(i), 'r') as f:
            for lineno, line in enumerate(f):
                if lineno % 2 == 0:
                    pid, text1 = line.strip().split('\t')
                    text1 = text1.strip()
                else:
                    text2 = line.strip()
                    texts.append(text1 + text2)
                    pids.append(pid)

    wf = open("./dataset/phd_qualified_seeds_inference_output_all.csv", 'w')
    for pid, text in tqdm(zip(pids, texts)):
        try:
            wf.writelines('{}\t{}\n'.format(pid, text))
            wf.flush()
        except Exception as e:
            print("===!!!Exception", e)
    wf.close()
