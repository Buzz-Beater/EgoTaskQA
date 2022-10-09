import json, nltk, sys
from pathlib import Path

base_data_dir = sys.argv[1]
with open(f'{base_data_dir}/tagged_qas.json', 'r') as f:
    tagged_qas = json.load(f)
    char_lst = ['|', ]
    for qa in tagged_qas:
        question = qa['question'].lower()[:-1]
        words = nltk.word_tokenize(question)
        # sentence = qa['question'].lower().replace(',', '').replace('?', '').replace('\'s', ' \'s')
        # words = sentence.split()
        for w in words:
            for c in list(w):
                if c in char_lst:
                    continue
                char_lst.append(c)
    with open(f'{base_data_dir}/char_vocab.txt', 'w') as outf:
        for ch in char_lst:
            outf.write(ch)
            outf.write('\n')