"""

    Created on 2022/10/9

    @author: Baoxiong Jia

"""
import json
import copy as cp
import random
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="direct", type=str,
                        help="split type")
    parser.add_argument("--base_dir", default='data/lemma-qa/', type=str,
                        help="default folder for original qas")
    args = parser.parse_args()
    return args


def avg_splitting(qas, ratios):
    results = {key : [] for key in ratios}
    sum_ratios = sum(ratios.values())
    typeans2lst = {}
    for qa in qas:
        typeans = qa["answer"]
        if typeans not in typeans2lst.keys():
            typeans2lst[typeans] = [cp.copy(qa)]
        else:
            typeans2lst[typeans].append(qa)
    for typeans, qa_lst in typeans2lst.items():
        random.shuffle(qa_lst)
        all_length =len(qa_lst)
        if all_length < sum_ratios:
            # split uniformly not following the original ratio
            if all_length >= len(ratios.keys()):
                cur_ratios = {x : 1 for x in ratios.keys()}
                cur_sum_ratios = len(ratios.keys())
            else:
                keys = random.sample(list(ratios.keys()), k=all_length)
                for k_idx, key in enumerate(keys):
                    results[key].append(qa_lst[k_idx])
                continue
        else:
            cur_ratios = ratios
            cur_sum_ratios = sum_ratios
        nums = sorted({key : int(all_length * val / cur_sum_ratios) for key, val in cur_ratios.items()}.items(), key=lambda x: x[0], reverse=True)
        start = 0
        for n_idx, (key, num) in enumerate(nums):
            if n_idx != len(nums) - 1:
                end = start + num
            else:
                end = max(all_length, start + num)
            results[key] += qa_lst[start : end]
            start = end
    assert sum([len(x) for x in results.values()]) == len(qas), f"not equally partitioned, missing {sum([len(x) for x in results.values()]) - len(qas)}"
    return results


def main(args):
    save_path = Path(args.base_dir + f"/{args.type}")
    save_path.mkdir(exist_ok=True, parents=True)
    with open(args.base_dir + '/vid_intervals.json', 'r') as f:
        vid_intervals = json.load(f)
    with (save_path / 'vid_intervals.json').open('w') as f:
        json.dump(vid_intervals, f, indent=4)
    with open(args.base_dir + '/tagged_qas.json', 'r') as f:
        vid_intervals = json.load(f)
    with (save_path / 'tagged_qas.json').open('w') as f:
        json.dump(vid_intervals, f, indent=4)
    with open(args.base_dir + '/tagged_qas.json', 'r') as f:
        tagged_qas = json.load(f)
        print(f"Generating {args.type} split")
        all_qas = []
        direct_qas = []
        indirect_qas = []
        for qa in tagged_qas:
            qa["reasoning_type"] = "$".join(qa["type"] + qa["category"] + [qa["semantic"], qa["structural"], qa["reference"]])
            if qa["reference"] == "direct":
                    direct_qas.append(cp.copy(qa))
            else:
                indirect_qas.append(cp.copy(qa))
            all_qas.append(cp.copy(qa))
        if args.type == "direct":
            random.shuffle(all_qas)
            ratios = {"train": 3, "test": 1, "val": 1}
            results = avg_splitting(all_qas, ratios)
            print({k : len(v) for k, v in results.items()})
            for key, qa_lst in results.items():
                with open(str(save_path) + f'/{key}_qas.json', 'w') as f:
                    json.dump(qa_lst, f)
        else:
            train_qas = direct_qas
            random.shuffle(indirect_qas)
            new_indirect_qas = []
            for qa in indirect_qas:
                if qa["program"].startswith("query([hoi], iterate_until("):
                    train_qas.append(qa)
                else:
                    new_indirect_qas.append(qa)
            ratios = {"test": 1, "val": 1}
            results = avg_splitting(new_indirect_qas, ratios)
            results["train"] = train_qas
            print({k : len(v) for k, v in results.items()})
            for key, qa_lst in results.items():
                with open(str(save_path) + f'/{key}_qas.json', 'w') as f:
                    json.dump(qa_lst, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)