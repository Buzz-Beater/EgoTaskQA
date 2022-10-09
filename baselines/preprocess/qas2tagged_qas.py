"""

    Created on 2022/10/9

    @author: Baoxiong Jia

"""

from glob import glob
import json
import argparse

import glob, os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="qas.json")
    parser.add_argument("--path", type=str, default="data/lemma-qa")
    args = parser.parse_args()
    return args


def load_video_paths(args):
    ori_file_path = args.path + "/" + args.file
    with open(ori_file_path) as f:
        qas = json.load(f)
        interval2video_id = {}
        video_id = 0
        question_id = 0
        for qa in qas:
            if qa['interval'] not in interval2video_id.keys():
                interval2video_id[qa['interval']] = video_id
                qa['video_id'] = video_id
                video_id += 1
            else:
                qa['video_id'] = interval2video_id[qa['interval']]

            qa['question_id'] = question_id
            question_id += 1

        print(f'writing to {args.path}/tagged_qas.json')
        with open(args.path + "/tagged_qas.json", 'w') as f:
            json.dump(qas, f)
    print('total num of qas:', len(qas))
    unique_qas = []
    existing_video_ids = []
    for qa in qas:
        if qa['video_id'] in existing_video_ids:
            continue
        else:
            existing_video_ids.append(qa['video_id'])
            unique_qas.append(qa)
    print('total num of video_id:', len(unique_qas))


if __name__ == '__main__':
    args = parse_args()
    load_video_paths(args)
