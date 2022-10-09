"""

    Created on 2022/5/12

    @author: Baoxiong Jia

"""

import re
import glob
import itertools
from pathlib import Path
from tqdm import tqdm

import utils.basic_utils as bu
from utils.log_utils import LOGGER


verb_templates = {
    "blend": "Blend {} using {}",
    "clean": "Clean with {}",
    "close": "Close {}",
    "cook": "Cook {} using {}",
    "cut": "Cut {} using {}",
    "drink": "Drink {} with {}",
    "eat": "Eat {} with {}",
    "fill": "Fill {} using {}",
    "get": "Get {} from {} using {}",
    "open": "Open {}",
    "play": "Play {} using {}",
    "point": "Point to {}",
    "pour": "Pour from {} into {} using {}",
    "put": "Put {} to {} using {}",
    "sit": "Sit down on {}",
    "sweep": "Sweep {} using {}",
    "switch": "Switch with {}",
    "throw": "Throw {} into {}",
    "turn-off": "Turn off {} with {}",
    "turn-on": "Turn on {} with {}",
    "watch": "Watch {}",
    "wash": "Wash {}",
    "work-on": "Work on {}",
    "stand-up": "stand-up"
}


# operations
def load_metadata(base_path):
    metadata_paths = [x for x in (Path(base_path) / "metadata").iterdir()]
    metadata = {x.stem : bu.load_json(x) for x in metadata_paths}
    metadata["intervals"] = bu.load_json(Path(base_path) / "action_intervals_w_tasks.json")
    return metadata


def load_generation_metadata(base_path, data_path):
    metadata_paths = [x for x in (Path(base_path) / "metadata").iterdir()]
    metadata = {x.stem : bu.load_json(x) for x in metadata_paths}
    metadata["intervals"] = bu.load_json(Path(base_path) / "per_act_intervals.json")
    if "lengths" not in metadata.keys():
        LOGGER.info("calculating lengths....")
        metadata["lengths"] = {}
        for vid, person in metadata["intervals"].items():
            for pid in person.keys():
                real_pid = "fpv1" if pid == "P1" else "fpv2"
                local_data_path = Path(data_path) / f"{vid}" / f"{real_pid}"
                image_paths = glob.glob(str(local_data_path / "img_[0-9]*.jpg"))
                if vid not in metadata["lengths"].keys():
                    metadata["lengths"][vid] = {pid : len(image_paths)}
                else:
                    metadata["lengths"][vid][pid] = len(image_paths)
        bu.save_json(metadata["lengths"], Path(base_path) / "metadata" / "lengths.json", save_pretty=True)
    if "hois" not in metadata.keys():
        LOGGER.info("generating hoi list....")
        # Regenerate per-frame annotation
        metadata["hois"] = set()
        for vid, v_intervals in tqdm(metadata["intervals"].items()):
            for pid, p_intervals in v_intervals.items():
                for interval in p_intervals:
                    interval = parse_interval(interval)
                    hoi, _ = parse_action(interval, metadata)
                    metadata["hois"].add(hoi)
        metadata["hois"].add("NULL")
        metadata["hois"] = list(metadata["hois"])
        bu.save_json(metadata["hois"], Path(base_path) / "metadata" / "hois.json", save_pretty=True)
    return metadata


def parse_action(interval, metadata):
    action, task = interval["name"], interval["task"]
    tasks = task.split(",") if "," in task else [task]
    action_id = -1
    action_verb = action.split(" ")[0].lower()
    for key, templ in verb_templates.items():
        verb = templ.split(" ")[0].lower()
        if action_verb == verb:
            if action_id == -1:
                action_id = metadata["actions"].index(key)
            else:
                action_verbs = " ".join(action.split(" ")[:2]).lower()
                verbs = " ".join(templ.split(" ")[:2]).lower()
                if action_verbs == verbs:
                    action_id = metadata["actions"].index(key)

    assert action_id != -1, "No verb found"
    obj_patterns = "\[[A-Za-z0-9\(\)\-,\ ]*\]"
    objs = re.findall(obj_patterns, action)
    objects = [x[1:-1].replace(" ", "").split(",") if "," in x else [x[1 : -1]] for x in objs]
    objects = [sorted(x) for x in objects]
    object_ids = [[metadata["objects"].index(x) for x in y] for y in objects]
    task_ids = [metadata["tasks"].index(x) for x in tasks]
    hoi = verb_templates[metadata["actions"][action_id]].format(*("[" + ",".join(x) + "]" for x in objects))
    return hoi, {"verb": action_id, "object": object_ids, "task": task_ids}


def parse_interval(interval):
    cols = ["name", "start", "end", "task"]
    result = {cols[x]: interval[x] for x in range(len(cols))}
    return result


def generate_segments(frames):
    def match_func(x):
        return ",".join([x[0], "$".join([str(a) for a in x[-1]])])
    all_segments = []
    all_segment_indices = []
    for idx, (_, g) in enumerate(itertools.groupby(frames, key=match_func)):
        all_segments.append(list(g))
        frame_info = all_segments[-1][0]
        vid_name, frame_id = frame_info[0], frame_info[1]
        all_segment_indices.append((vid_name, frame_id, len(all_segments[-1])))
    return all_segments, all_segment_indices


def generate_annotations(metadata, old_annotation_path, new_save_path):
    old_annotations = bu.load_pickle(old_annotation_path / "rec_fpv_frames.p")
    new_annotations = []
    for frame in tqdm(old_annotations):
        vid, pid = frame[0].split("|")
        frame_id = frame[1]
        if vid not in metadata["intervals"].keys():
            print(f"Skipping video {vid}")
            continue
        act_intervals = metadata["intervals"][vid][pid]
        verb_ids = []
        object_list = []
        task_list = []
        hoi_list = []
        for interval in act_intervals:
            interval = parse_interval(interval)
            if frame_id < interval["end"] and frame_id >= interval["start"]:
                hoi, ids = parse_action(interval, metadata)
                verb_ids.append(ids["verb"])
                object_list.append(ids["object"])
                task_list.append(ids["task"])
                hoi_list.append(metadata["hois"].index(hoi))
        if len(hoi_list) == 0:
            hoi_list.append(metadata["hois"].index("NULL"))
        new_annotations.append(frame[:5] + [task_list, hoi_list])
    _, segment_indices = generate_segments(new_annotations)
    bu.save_pickle(segment_indices, new_save_path / "refined_rec_fpv_segments_indices.p")
    bu.save_pickle(new_annotations, new_save_path / "refined_rec_fpv_frames.p")


def hoi2verb(hoi):
    found = None
    verb = hoi.split(" ")[0].lower()
    for key, tmpl in verb_templates.items():
        tmpl_verb = tmpl.split(" ")[0].lower()
        if verb == tmpl_verb:
            if found is not None:
                found = key
            else:
                verbs = " ".join(hoi.split(" ")[:2]).lower()
                tmpl_verbs = " ".join(tmpl.split(" ")[:2]).lower()
                if verbs == tmpl_verbs:
                    found = key
    return found


def hoi2objects(hoi):
    obj_patterns = "\[[A-Za-z0-9\(\)\-,\ ]*\]"
    objs = re.findall(obj_patterns, hoi)
    objects = [x[1:-1].replace(" ", "").split(",") if "," in x else [x[1: -1]] for x in objs]
    objects = [sorted(x) for x in objects]
    return objects


def get_version(hoi, version="past"):
    verb = hoi2verb(hoi)
    # TODO: write version file here
    pass


if __name__ == "__main__":
    base_path = "/home/baoxiong/Datasets/LEMMA/annotations/new_annotations"
    data_path = "/home/baoxiong/Datasets/LEMMA/videos"
    old_annotation_path = Path(base_path) / "annotations"
    new_save_path = old_annotation_path
    metadata = load_generation_metadata(base_path, data_path)
    generate_annotations(metadata, old_annotation_path, new_save_path)