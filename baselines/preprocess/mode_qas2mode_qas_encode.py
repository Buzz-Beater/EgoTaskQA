import nltk
import json, sys


base_data_dir = sys.argv[1]
modes = ['train' ,'test', 'val']
for mode in modes:
    input_file = f'{base_data_dir}/{mode}_qas.json'
    # # output: {mode}_qas_encode.json, answer_set.txt, vocab.txt
    output_file = f'{base_data_dir}/{mode}_qas_encode.json'

    with open(f'{base_data_dir}/lemma-qa_vocab.json', 'r') as lemma_vocab_f:
        input_vocab = json.load(lemma_vocab_f)
        answer_set_lst = []
        # # generate answer_set
        for ans in input_vocab['answer_token_to_idx']:
            if ans not in answer_set_lst:
                answer_set_lst.append(ans)
        with open(f'{base_data_dir}/answer_set.txt', 'w') as answerset_f:
            for ans in answer_set_lst:
                answerset_f.write(ans)
                answerset_f.write('\n')
        
        # # generate vocab_set
        max_word_len = 1
        vocab_lst = []
        for word in input_vocab['question_token_to_idx']:
            if word not in vocab_lst:
                vocab_lst.append(word)
        with open(f'{base_data_dir}/vocab.txt', 'w') as vocab_f:
            for word in vocab_lst:
                max_word_len = max(max_word_len, len(word))
                vocab_f.write(word)
                vocab_f.write('\n')
        
        print(">>> maxwordlen:", max_word_len)

        max_sentence_len = 1
        with open(input_file, 'r') as f:
            qas = json.load(f)
            print('total qas:', mode, len(qas))
            for qa in qas:
                
                # question_word_lst = qa['question'][:-1].split(' ') # #去掉标点符号
                encoded_q = []
                question = qa['question'].lower()[:-1]
                question_word_lst = nltk.word_tokenize(question)
                max_sentence_len = max(max_sentence_len, len(question_word_lst))
                
                for word in question_word_lst:
                    word = word.lower()
                    if word not in vocab_lst:
                        encoded_q.append(str(vocab_lst.index('<UNK>')))
                        # import pdb; pdb.set_trace()
                        print(f'questionword of {input_file}:{word} not in vocab_lst')
                    else:
                        encoded_q.append(str(vocab_lst.index(word)))
                encoded_q = ' '.join(encoded_q)
                qa['question_encode'] = encoded_q
                if mode == 'train':
                    qa['answer_encode'] = str(answer_set_lst.index(qa['answer'].lower()))
                else:
                    if qa['answer'] in answer_set_lst:
                        qa['answer_encode'] = str(answer_set_lst.index(qa['answer'].lower()))
                    else:
                        # print(mode, qa['answer'])
                        qa['answer_encode'] = str(answer_set_lst.index("<UNK1>"))
            with open(output_file, 'w') as outf:
                json.dump(qas, outf)
        
        print('max sentence len:', max_sentence_len)

