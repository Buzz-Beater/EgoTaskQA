"""

    Created on 2022/5/12

    @author: Baoxiong Jia

"""

import re
import argparse
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import copy as cp

import utils.basic_utils as bu
from utils.log_utils import LOGGER
from generate.executor import execute
import generate.process_anno as pu
import generate.qa_formatting as qu
import utils.anno_utils as au
import utils.video_utils as vu


def load_annotation(anno_path, metadata_path, data_path):
    annotations = bu.load_json(anno_path)
    metadata = au.load_generation_metadata(metadata_path, data_path)
    return annotations, metadata


def fill_template(templates, sample_space, interval_anno, interval_id, causal_trace):
    def verb2ing(string):
        if string.endswith("e"):
            return string[:-1] + "ing"
        elif string.endswith("t") and string not in ["eat", "point"]:
            return string + "ting"
        else:
            return string + "ing"
    def dfs(template, all_scopes, interval_anno, results=None):
        if len(all_scopes) == 0:
            results.append(cp.deepcopy(template))
            return
        param_scope = all_scopes[0]
        param_name, scopes = param_scope
        # No valid entities for this part any more
        if len(scopes) == 0:
            return
        for s_idx in range(len(scopes)):
            ori_selected_val = scopes.pop(s_idx)
            selected_type = None
            new_template = cp.deepcopy(template)
            if param_name.startswith("{fv"):
                selected_val, selected_type = ori_selected_val.split("&")
                if selected_val == "unknown":
                    return
            else:
                selected_val = ori_selected_val
            if param_name.startswith("{n"):
                vals = selected_val.split(" ")
                vals[0] = verb2ing(vals[0])
                selected_val_ing = " ".join(vals)
                new_template["question"] = new_template["question"].replace(param_name, f"the action {selected_val_ing.lower()}")
                new_template["program"] = new_template["program"].replace(param_name, f"'{selected_val.lower()}'")
            else:
                new_template["question"] = new_template["question"].replace(param_name, f"{selected_val.lower()}")
                new_template["program"] = new_template["program"].replace(param_name, f"'{selected_val.lower()}'")
            if selected_type is not None:
                 new_template["program"] = new_template["program"].replace(param_name.replace("v", ""), f"'{selected_type.lower()}'")
            
            new_all_scopes = cp.deepcopy(all_scopes)
            pop_idx = [(x, set()) for x, _ in new_all_scopes]
            for new_idx, (n_param_name, n_scopes) in enumerate(new_all_scopes):
                param_scope_id = "".join([x if x.isdigit() else "" for x in param_name])
                new_scope_id = "".join([x if x.isdigit() else "" for x in n_param_name])
                if param_scope_id == new_scope_id:
                    continue
                for n_scope_idx, n_scope in enumerate(n_scopes):
                    # For following attributes that has a different index, remove this fluent
                    if n_param_name.startswith("{fv"):
                        s_selected_val, s_selected_type = n_scope.split("&")
                        if param_name.startswith("{fv"):
                            if s_selected_val == selected_val:
                                pop_idx[new_idx][1].add(n_scope_idx)
                        # f was mentioned, remove all related states in fvs
                        elif param_name.startswith("{f"):
                            if s_selected_type == selected_val:
                                pop_idx[new_idx][1].add(n_scope_idx)
                    elif n_param_name.startswith("{f"):
                        s_selected_type = n_scope
                        if param_name.startswith("{fv"):
                            if selected_type == s_selected_type:
                                pop_idx[new_idx][1].add(new_idx)
                        elif param_name.startswith("{f"):
                            if s_selected_type == selected_val:
                                pop_idx[new_idx][1].add(new_idx)
                    elif n_param_name.replace(new_scope_id, "") == param_name.replace(param_scope_id, ""):
                        s_selected_type = n_scope
                        if s_selected_type == selected_val:
                            pop_idx[new_idx][1].add(n_scope_idx)
            filtered_all_scopes = []
            for new_idx, (n_param_name, n_scopes) in enumerate(new_all_scopes):
                filter_scopes = []
                for n_scope_idx, n_scope in enumerate(n_scopes):
                    if n_scope_idx not in pop_idx[new_idx][1]:
                        filter_scopes.append(n_scope)
                filtered_all_scopes.append((n_param_name, filter_scopes))
            dfs(new_template, filtered_all_scopes[1:], interval_anno, results=results)
            
            # dfs(new_template, all_scopes[1:], interval_anno, results=results)
            scopes.insert(s_idx, ori_selected_val)
            del new_template
    qa_pairs = []
    types = ["before", "after"]
    for tmpl in templates:
        question = tmpl["question"]
        pattern = "\{[A-Za-z0-9]*\}"
        all_params = sorted([x for x in re.findall(pattern, question)])
        all_scopes = []
        for param in all_params:
            if param[1] == "o":
                if param[2] == "h":
                    all_scopes.append((param, sample_space["ohoi"]))
                elif param[2] == "a":
                    all_scopes.append((param, sample_space["oaction"]))
                else:
                    all_scopes.append((param, sample_space["obj"]))
            if param[1] == "r":
                all_scopes.append((param, sample_space["relationship"]))
            if param[1] == "f":
                if param[2] == "v":
                    all_scopes.append((param, sample_space["state"]))
                else:
                    all_scopes.append((param, sample_space["change"]))
            if param[1] == "h":
                all_scopes.append((param, sample_space["hoi"]))
            if param[1] == "a":
                all_scopes.append((param, sample_space["action"]))
            if param[1] == "t":
                all_scopes.append((param, types))
            if param[1] == "i":
                if param[2] == "h":
                    all_scopes.append((param, sample_space["hoi"]))
                else:
                    all_scopes.append((param, sample_space["action"]))
            if param[1] == "n":
                if param[2] == "h":
                    all_scopes.append((param, sample_space["hoi"]))
                else:
                    all_scopes.append((param, sample_space["action"]))
        results = []
        dfs(tmpl, all_scopes, interval_anno, results=results)
        for r_idx, result in enumerate(results):
            answer = execute(result["program"], interval_anno, causal_trace)
            if answer is not None:
                result["answer"] = answer
                result["interval"] = interval_id
                result = qu.post_process(result)
                if result is not None:
                    qa_pairs += cp.deepcopy(qu.process_obj_vis(result, interval_anno, causal_trace, args.vis_num))
    return qa_pairs


def generate_samplespace(interval_anno):
    sample_space = {"obj": [], "relationship":[], "change":[], "state":[], "hoi": [], "action": [], "ohoi": [],
                    "oaction": []}
    all_interval_anno = []
    for anno in interval_anno:
        all_interval_anno.append(anno)
        if anno["others"] is not None:
            all_interval_anno += anno["others"]
    for interval in all_interval_anno:
        sample_space["hoi"].append(interval["hoi"])
        sample_space["action"].append(interval["action"])
        if "is_multi" not in interval.keys():
            sample_space["ohoi"].append(interval["hoi"])
            sample_space["oaction"].append(interval["action"])
        if not interval["state"].empty:
            sample_space["obj"] += list(interval["state"]["obj"].values)
        if not interval["relationship"].empty:
            sample_space["relationship"].append(interval["relationship"]["relationship"].values[0])
        columns = (list(interval["state"].columns) if not interval["state"].empty else []) + \
                  (list(interval["change"]["change"].values) if not interval["change"].empty else [])
        for fluent_name in columns:
            if fluent_name not in [
                "obj", "type", "before", "after", "visibility to me",
                "visibility to the other person", "visibility"]:
                sample_space["change"].append(fluent_name)
        for fluent_name in sample_space["change"]:
            if fluent_name != "spatial relationships":
                if fluent_name not in interval["state"]:
                    continue
                sample_space["state"] += [x + "&" + fluent_name for x in list(interval["state"][fluent_name].values)]
            else:
                if "change" not in interval["change"]:
                    continue
                spatial_relationships = interval["change"].loc[interval["change"]["change"] == "spatial"]
                if not spatial_relationships.empty:
                    sample_space["state"] += [x + "&" + fluent_name for x in list(spatial_relationships["before"].values)]
                    sample_space["state"] += [x + "&" + fluent_name for x in list(spatial_relationships["after"].values)]
    for key, val in sample_space.items():
        sample_space[key] = sorted(list(set(val)))
    return sample_space


def generate(interval, vid_annotations, metadata, args):
    clip_interval, _, pred_start = interval
    interval_id = "|".join(clip_interval.split("|")[0 : 3] + [str(pred_start)])
    interval_anno = pu.gather_annotation(interval, vid_annotations, metadata)
    causal_trace = pu.gather_causal_trace(interval_anno, vid_annotations, metadata)
    interval_anno = pu.reformat(interval_anno)
    sample_space = generate_samplespace(interval_anno)
    templates = bu.load_json(args.tmpl_path)
    results = fill_template(templates, sample_space, interval_anno, interval_id, causal_trace)
    new_results = []
    for x in results:
        new_results.append(x)
    return new_results


def main(args):
    annotations, metadata = load_annotation(args.anno_path, args.metadata_path, args.data_path)
    all_clip_intervals = vu.generate_intervals(metadata, prediction_window_length=args.pred_window,
                                                min_length=args.min_duration)
    all_vid_intervals = set()
    for clip_interval, _, pred_start in all_clip_intervals:
        interval_id = "|".join(clip_interval.split("|")[0 : 3] + [str(pred_start)])
        all_vid_intervals.add(interval_id)
    print(len(all_vid_intervals))
    bu.save_json(sorted(list(all_vid_intervals)), args.save_file.parent / "vid_intervals.json")
    LOGGER.info("Intervals loaded, generating questions.....")
    if args.dist:
        LOGGER.info("Using multiparallel")
        all_bar_funcs = {'tqdm': lambda args: lambda x: tqdm(x, **args)}
        def ParallelExecutor(use_bar='tqdm', **joblib_args):
            def aprun(bar=use_bar, **tq_args):
                def tmp(op_iter):
                    if str(bar) in all_bar_funcs.keys():
                        bar_func = all_bar_funcs[str(bar)](tq_args)
                    else:
                        raise ValueError("Value %s not supported as bar type"%bar)
                    return Parallel(**joblib_args)(bar_func(op_iter))
                return tmp
            return aprun
        with parallel_backend("multiprocessing", n_jobs=100):
            aprun = ParallelExecutor(n_jobs=100)
            all_qa_pairs = aprun(total=len(all_clip_intervals))(
                delayed(generate)(interval, annotations, metadata, args) for interval in all_clip_intervals
            )
        LOGGER.info("parallel execution finished")
        final = []
        for qa_pairs in all_qa_pairs:
            final += qa_pairs
        all_qa_pairs = final
    else:
        all_qa_pairs = []
        for idx, interval in enumerate(tqdm(all_clip_intervals)):
            all_qa_pairs += generate(interval, annotations, metadata, args)
            if (idx + 1) % args.log_interval == 0:
                LOGGER.info(f"Current total questions {len(all_qa_pairs)}")
    LOGGER.info(f"Generating {len(all_qa_pairs)} of questions")
    bu.save_json(all_qa_pairs, args.save_file)


def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')
    parser = argparse.ArgumentParser("QA generation arguments")
    parser.add_argument("--min_duration", type=int, default=15,
                        help="minimum duration for each video.")
    parser.add_argument("--pred_window", type=int, default=3,
                        help="number of action intervals to look ahead in prediciton.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="log interval for periodic information.")
    parser.add_argument("--dist", type=str2bool, default=False,
                        help="Use multiprocessing for generating questions or not.")
    parser.add_argument("--file_base", type=str, default="/mnt/hdd/data/EgoTaskQA",
                        help="base file path for question templates")
    parser.add_argument("--file", type=str, default="direct.json",
                        help="template file")
    parser.add_argument("--save_file", type=str, default="/mnt/hdd/data/EgoTaskQA/qas.json", help="save_file")
    parser.add_argument("--mask_num", type=int, default=1, help="number of objects to mask")
    parser.add_argument("--anno_path", type=str, default="/mnt/hdd/data/EgoTaskQA/parsed.json",
                        help="the default file path for world state annotations")
    args = parser.parse_args()
    args.vis_num = args.mask_num
    project_root = Path("/home/baoxiong/Datasets/LEMMA")
    args.data_path = project_root / "videos"
    args.metadata_path = project_root / "annotations" / "new_annotations"

    args.anno_path = Path(args.anno_path)
    args.save_file = Path(args.save_file)
    args.tmpl_path = Path(args.file_base) / args.file
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)