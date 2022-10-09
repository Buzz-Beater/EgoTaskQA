import argparse
import json
import copy as cp
import random

# # reasoning_types = ["0", "1", "2", "3"]

# binary_reasoning_types = ["0"]

def balance_binary_qa(binary_qa_lst, reason_type):
    '''
    :param qa_lst: list of questions and answers from the same reasoning type
    '''
    answer_frequency = {"yes": 0, "no": 0}
    for qa in binary_qa_lst:
        answer_frequency[qa["answer"]] += 1
    if len(binary_qa_lst) < 10:
        return []
    
    yes_qa_lst = [qa for qa in binary_qa_lst if qa["answer"] == "yes"]
    no_qa_lst = [qa for qa in binary_qa_lst if qa["answer"] == "no"]
    
    if answer_frequency["yes"] > answer_frequency["no"]:
        remove_num = answer_frequency["yes"] - answer_frequency["no"]
        random.shuffle(yes_qa_lst)
        del yes_qa_lst[:remove_num]
    else:
        remove_num = answer_frequency["no"] - answer_frequency["yes"]
        random.shuffle(no_qa_lst)
        del no_qa_lst[:remove_num]
    
    binary_qa_lst = yes_qa_lst + no_qa_lst
    return binary_qa_lst


def balance_open_qa(open_qa_lst, args, curkey):
    '''
    :param qa_lst: list of questions and answers
    '''
    answer_frequency = {} 
    total_num_of_qa = len(open_qa_lst)
    random.shuffle(open_qa_lst)

    for qa in open_qa_lst:
        if qa["answer"] not in answer_frequency:
            answer_frequency[qa["answer"]] = 1
        else:
            answer_frequency[qa["answer"]] += 1
    for qa in open_qa_lst:
        qa['frequency'] = answer_frequency[qa["answer"]]
    open_qa_lst.sort(key=lambda x: x['frequency'], reverse=True) # # only need to sort open_qa_lst once
    # answer_frequency.sort(key=lambda x: x[1], reverse=True)
    answer_frequency = sorted(answer_frequency.items(), key=lambda x: x[1], reverse=True)
    answer_frequency = [list(item) for item in answer_frequency]

    # Check if there are various possibilities for the answer
    if len(answer_frequency) < 10:
        print(open_qa_lst[0]["program"])
    alpha = args.alpha
    beta = args.beta
    for i in range(len(answer_frequency)): # # at most iterate len(answer_frequency) times  
        # # tmp_sum: 计算前20%的answer type 占有多少qa
        tmp_sum = 0
        for j, (ans, freq) in enumerate(answer_frequency):
            if j > len(answer_frequency) * alpha - 1:
                break
            tmp_sum += freq
            
        if tmp_sum >= total_num_of_qa * beta:
            assert i+1 < len(answer_frequency)

            # # remove the most frequent qa
            remove_num_per_answer = answer_frequency[i][1] - answer_frequency[i+1][1]
            
            total_num_of_qa -= remove_num_per_answer * (i + 1)
            for j, (key, value) in enumerate(answer_frequency):
                if j == i + 1:
                    break
                answer_frequency[j][1] -= remove_num_per_answer    

            for j in range(i + 1):
                start_idx = j * answer_frequency[0][1]
                open_qa_lst = open_qa_lst[0:start_idx] + open_qa_lst[start_idx + remove_num_per_answer:]
        else: # # 满足前20%的answer type 占有的qa不到30%
            break
    for open_qa in open_qa_lst:
        del open_qa['frequency']
    return open_qa_lst

    
def balance_answer_distribution(qa_lst, args):
    '''
    :param qa_lst: list of questions and answers
    :return: list of questions and answers
    balancing rule: 
        1. for binary questions, remove questions to make both answer equally plausible
        2. for open questions, delete questions so that 20% of answers represent at most 30% of all questions
    '''
    print(len(qa_lst))
    binary_qa_dct = {} # # key: reasoning type, value: list of questions and answers
    open_qa_dct = {} # # key: reasoning type, value: list of questions and answers
    for qa in qa_lst:
        if "sturctural" in qa.keys():
            qa["structural"] = qa["sturctural"]
        qa["reasoning_type"] = "$".join(qa["type"] + qa["category"] + [qa["semantic"], qa["structural"], str(qa["step"])])
        if qa["answer"] in ['yes', 'no']:
            if qa["reasoning_type"] not in binary_qa_dct:
                binary_qa_dct[qa["reasoning_type"]] = [qa]
            else:
                binary_qa_dct[qa["reasoning_type"]].append(qa)
        else:
            if qa["reasoning_type"] not in open_qa_dct:
                open_qa_dct[qa["reasoning_type"]] = [qa]
            else:
                open_qa_dct[qa["reasoning_type"]].append(qa)
    before_frequency = {}
    for reasoning_type, qas in open_qa_dct.items():
        reasoning_type = "$".join(reasoning_type.split("$")[:-1])
        if reasoning_type not in before_frequency.keys():
            before_frequency[reasoning_type] = {}
        for qa in qas:
            answer = qa["answer"]
            if answer not in before_frequency[reasoning_type].keys():
                before_frequency[reasoning_type][answer] = 1
            else:
                before_frequency[reasoning_type][answer] += 1
    
    # # balance binary questions: 
    for key, value in binary_qa_dct.items():
        binary_qa_dct[key] = balance_binary_qa(value, key)
    # # balance open questions:
    for key, value in open_qa_dct.items():
        open_qa_dct[key] = balance_open_qa(value, args, curkey=key)
    # # merge binary and open questions:
    qa_lst = []
    for k, v in binary_qa_dct.items():
        print(k, len(v))
    
    for k, v in open_qa_dct.items():
        print(k, len(v))

    print('-------')
    for key, value in binary_qa_dct.items():
        qa_lst += value
    print("binary_num", len(qa_lst))
    for key, value in open_qa_dct.items():
        qa_lst += value
    print("total_qas before balance by type", len(qa_lst))
    return qa_lst, cp.deepcopy(before_frequency)

def balance_by_type(qa_lst, type_ratio_dct):

    typq_qa_dct = {}
    for _type in type_ratio_dct.keys():
        typq_qa_dct[_type] = []
    for qa in qa_lst:
        for _type in type_ratio_dct.keys():
            if _type in qa['reasoning_type']:
                typq_qa_dct[_type].append(qa)
                break

    _min = min([len(typq_qa_dct[k]) // type_ratio_dct[k] for k in typq_qa_dct.keys()])
    for _type in typq_qa_dct.keys():
        random.shuffle(typq_qa_dct[_type])
        num_to_remove = len(typq_qa_dct[_type]) - _min * type_ratio_dct[_type]
        del typq_qa_dct[_type][:num_to_remove]

    qa_lst = []
    for _type in typq_qa_dct.keys():
        qa_lst += typq_qa_dct[_type]

    overall_ans_dist = {}
    overall_type_dist = {}
    for qa in qa_lst:
        if qa["reasoning_type"] not in overall_type_dist.keys():
            overall_type_dist[qa["reasoning_type"]] = 1
        else:
            overall_type_dist[qa["reasoning_type"]] += 1
        if qa["answer"] not in overall_ans_dist.keys():
            overall_ans_dist[qa["answer"]] = 1
        else:
            overall_ans_dist[qa["answer"]] += 1
    ans_dist = sorted(overall_ans_dist.items(), key=lambda x: x[1], reverse=True)
    type_dist = sorted(overall_type_dist.items(), key=lambda x: x[1], reverse=True)
    # print(ans_dist)
    # print(type_dist)
    print(f"final_qas {len(qa_lst)}")
    return qa_lst


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/home/baoxiong/Desktop/ori", type=str, help="file_path")
    parser.add_argument("--file", default="qas_merge_ori.json", type=str, help="input qa need balancing")
    parser.add_argument("--save_file", default="balance_qa_1:2_0.20.33_merge_act_ori_test.json", type=str, help="output balanced qa")
    parser.add_argument("--ans_dist", default="ans_dist.json", type=str)
    parser.add_argument("--alpha", default=0.2, type=float, help="percentage of answer types")
    parser.add_argument("--beta", default=0.33, type=float, help="percentage of questions")
    args = parser.parse_args()
    args.file = args.path + "/" + args.file
    args.save_file = args.path + "/" + args.save_file
    args.ans_dist = args.path + "/" + args.ans_dist
    return args


def vis_stats(qa_lst, msg):
    balance_ans_freq = {}
    for qa in qa_lst:
        qa["reasoning_type"] = "$".join(qa["type"] + qa["category"] + [qa["semantic"], qa["structural"]])
        for r_type in qa["reasoning_type"].split("$"):
            if r_type not in balance_ans_freq.keys():
                balance_ans_freq[r_type] = 1
            else:
                balance_ans_freq[r_type] += 1
    print(msg)
    print(balance_ans_freq)



# # balance answer distributation per reasoning type
if __name__ == '__main__':
    args = parse_args()
    input_file = args.file
    output_file = args.save_file

    with open(input_file, 'r') as f:
        qa_lst = json.load(f)
        vis_stats(qa_lst, "Before balancing stats")

        qa_lst, before_balance_ans_dist = balance_answer_distribution(qa_lst, args)
        all_reason_types = set([qa["reasoning_type"] for qa in qa_lst])
        qa_lst = balance_by_type(qa_lst, {"verify": 1, "query": 2})
        vis_stats(qa_lst, "After balancing stats")
        # qa_lst = balance_by_type(qa_lst, {"explanatory": 1, "descriptive": 1, "counterfactual": 1, "prediction": 1})
        freq = {}
        q_type_freq = {}
        q_scope_freq = {}
        after_balance_ans_dist = {}
        for qa in qa_lst:
            q_scope = "$".join(qa["category"])
            q_type = "$".join(qa["type"])
            if q_scope not in q_scope_freq.keys():
                q_scope_freq[q_scope] = 1
            else:
                q_scope_freq[q_scope] += 1
            if q_type not in q_type_freq.keys():
                q_type_freq[q_type] = 1
            else:
                q_type_freq[q_type] += 1
            # reasoning_type = qa["reasoning_type"].split("$")[:-1]
            reasoning_type = qa["reasoning_type"].split("$")
            type_str = "$".join(reasoning_type)
            reasoning_type.append(qa["reference"])
            if qa["answer"] not in ["yes", "no"]:
                if type_str not in after_balance_ans_dist.keys():
                    after_balance_ans_dist[type_str] = {}
                
                if qa["answer"] not in after_balance_ans_dist[type_str].keys():
                    after_balance_ans_dist[type_str][qa["answer"]] = 1
                else:
                    after_balance_ans_dist[type_str][qa["answer"]] += 1
            
            for r_type in reasoning_type:
                if r_type not in freq.keys():
                    freq[r_type] = 1
                else:
                    freq[r_type] += 1
            qa["reasoning_type"] = reasoning_type
        freq = {k : v / len(qa_lst) for k,v in freq.items()}
        q_scope_freq = {k : v / len(qa_lst) for k, v in q_scope_freq.items()}
        q_type_freq = {k : v / len(qa_lst) for k, v in q_type_freq.items()}
        print(freq)
        print(q_scope_freq)
        print(q_type_freq)
        
        results = {}
        for r_type in after_balance_ans_dist.keys():
            before_dist = sorted(before_balance_ans_dist[r_type].items(), key=lambda x : x[1], reverse=True)
            after_dist = sorted(after_balance_ans_dist[r_type].items(), key=lambda x : x[1], reverse=True)
            max_len = max(len(before_dist), len(after_dist))
            before = [x[1] for x in before_dist]
            after = [x[1] for x in after_dist]
            if len(before) < max_len:
                before += [0 for i in range(max_len - len(before))]
            if len(after) < max_len:
                after += [0 for i in range(max_len - len(after))]
            results[r_type] = [before, after]
        results = sorted(results.items(), key=lambda x : len(x[1][0]), reverse=True)
        with open(args.ans_dist, "w") as f:
            json.dump(results, f, indent=4)
        with open(output_file, 'w') as f:
            json.dump(qa_lst, f, indent=4)
            
    