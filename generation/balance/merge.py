import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/mnt/hdd/data/EgoTaskQA/qas_dir_act.json", type=str)
    parser.add_argument("--ind", default="/mnt/hdd/data/EgoTaskQA/qas_ind_act_obj.json", type=str)
    parser.add_argument("--save", default="/mnt/hdd/data/EgoTaskQA/qas_merged_act.json", type=str)
    args = parser.parse_args()
    return args


def main(args):
    with open(args.dir, "r") as f:
        dir_qas = json.load(f)
    with open(args.ind, "r") as f:
        ind_qas = json.load(f)
    merged_qas = []
    new_dir_qas = []
    new_ind_qas = []
    for qa in dir_qas:
        qa["reference"] = "direct"
        # del qa["step"]
        merged_qas.append(qa)
        new_dir_qas.append(qa)
    print("Direct", len(merged_qas))
    for qa in ind_qas:
        qa["reference"] = "indirect"
        # del qa["step"]
        merged_qas.append(qa)
        new_ind_qas.append(qa)
    with open(args.save, "w") as f:
        json.dump(merged_qas, f, indent=4)
    print("Total", len(merged_qas)) 
    

if __name__ == "__main__":
    args = parse_args()
    main(args)