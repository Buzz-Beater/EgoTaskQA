"""

    Created on 2022/5/12

    @author: Baoxiong Jia

"""

import re
import itertools
import copy as cp

import pandas as pd

import generate.qa_formatting as qu


def gather_simultaneous(interval_clip_id, vid, pid, pred_start, metadata, annotations):
    _, start1, end1 = interval_clip_id.split("|")
    o_all_intervals = metadata["intervals"][vid][pid]
    vid_anno = annotations[vid]["all_segments"]
    intersecting_intervals = []
    for interval in o_all_intervals:
        start2, end2 = interval[1], interval[2]
        if (int(start1) >= int(start2) and int(start1) < int(end2)) or \
            (int(end1) >= int(start2) and int(end1) < int(end2)) or \
            (int(start2) >= int(start1) and int(start2) < int(end1)) or \
            (int (end2) >= int(start1) and int(end2) < int(end1)):
            intersecting_intervals.append(interval)
    o_clip_annos = []
    for interval in intersecting_intervals:
        interval_clip_id = "|".join([interval[0], str(interval[1]), str(interval[2])])
        found = False
        for block in vid_anno:
            header, _, segments = block
            b_idx, b_pid, b_hois, b_start, b_end = header
            b_clip_id = "|".join([b_hois[0], str(b_start), str(b_end)])
            if b_clip_id == interval_clip_id and b_pid == pid:
                if not found:
                    interval_vid = f"{vid}@{pid}" + "|" + interval_clip_id
                    o_clip_anno = {
                        "hoi": interval[0],
                        "ori_hoi": interval[0],
                        "task": interval[3],
                        "action": qu.anonymize(interval[0], pid),
                        "interval_id": interval_vid,
                        "person": query_person([qu.compress_anno(x) for x in segments], interval_vid),
                        "relationship": query_relation([qu.compress_anno(x) for x in segments], interval_vid),
                        "state": query_state([qu.compress_anno(x) for x in segments], interval_vid),
                        "change": query_change([qu.compress_anno(x) for x in segments], interval_vid),
                        "executable": "yes",
                        "is_pred": False if pred_start > interval[1] else True,
                    }
                    if not o_clip_anno["person"].empty:
                        o_clip_anno["aware"] = "yes" if not \
                            o_clip_anno["person"][o_clip_anno["person"]["rel"] == "knows-the-other-person-action"].empty \
                            else "no"
                    else:
                        o_clip_anno["aware"] = "no"
                    o_clip_annos.append(o_clip_anno)
                else:
                    assert False, f"There should be a 1-to-1 correspondence for {vid}:{b_clip_id}"
    return o_clip_annos


def gather_annotation(interval, annotations, metadata, collect_both=False):
    """
        Gather annotation as well as causal trace as dictionary
    """
    _, action_ids, pred_start = interval
    vid, pid, aids = action_ids.split("|")
    aid_start, aid_end = aids.split("-")

    vid_anno = annotations[vid]["all_segments"]
    obj_chain = annotations[vid]["object_chains"]
    intervals = metadata["intervals"][vid][pid][int(aid_start) : int(aid_end) + 1]
    is_multi = True if len(metadata["intervals"][vid].keys()) > 1 else False
    clip_annos = []
    for i_id, interval in enumerate(intervals):
        interval_clip_id = "|".join([interval[0], str(interval[1]), str(interval[2])])
        found = False
        for block in vid_anno:
            header, _, segments = block
            b_idx, b_pid, b_hois, b_start, b_end = header
            b_clip_id = "|".join([b_hois[0], str(b_start), str(b_end)])    
            if b_clip_id == interval_clip_id and b_pid == pid:
                if not found:
                    found = True
                    # find corresponding annotation from the other person
                    if is_multi:
                        o_pid = "P1" if pid == "P2" else "P2"
                        others_simultaneous_anno = gather_simultaneous(interval_clip_id, vid, o_pid,
                                                                       pred_start, metadata, annotations)
                    else:
                        others_simultaneous_anno = None
                    interval_vid = f"{vid}@{pid}" + "|" + interval_clip_id
                    clip_anno = {
                            "hoi": interval[0],
                            "ori_hoi": interval[0],
                            "action": qu.anonymize(interval[0], pid=pid),
                            "task": interval[3],
                            "interval_id": interval_vid,
                            "person": query_person([qu.compress_anno(x) for x in segments], interval_vid),
                            "relationship": query_relation([qu.compress_anno(x) for x in segments], interval_vid),
                            "state": query_state([qu.compress_anno(x) for x in segments], interval_vid),
                            "change": query_change([qu.compress_anno(x) for x in segments], interval_vid),
                            "others": others_simultaneous_anno,
                            "executable": "yes",
                            "is_multi": "yes" if is_multi else "no",
                            "is_pred": False if pred_start > interval[1] else True,
                    }
                    if not clip_anno["person"].empty:
                        clip_anno["aware"] = "yes" if not \
                            clip_anno["person"][clip_anno["person"]["rel"] == "knows-the-other-person-action"].empty \
                            else "no"
                    else:
                        clip_anno["aware"] = "no"
                    clip_annos.append(clip_anno)
                else:
                    assert False, f"There should be a 1-to-1 correspondence for {vid}:{b_clip_id}"
        if not found:
            # LOGGER.error(f"Haven't found for {vid}:{interval_clip_id}")
            pass
    return clip_annos


def reformat(intervals_data):
    for idx, interval in enumerate(intervals_data):
        pid = interval["interval_id"].split("|")[0].split("@")[1]
        o_pid = "P2" if pid == "P1" else "P1"
        intervals_data[idx]["hoi"] = qu.pretty_hoi(interval["hoi"], pid)
        if intervals_data[idx]["others"] is not None:
            for o_idx, o_interval in enumerate(intervals_data[idx]["others"]):
                o_hoi = intervals_data[idx]["others"][o_idx]["hoi"]
                intervals_data[idx]["others"][o_idx]["hoi"] = qu.pretty_hoi(o_hoi, o_pid)
    return intervals_data


def gather_causal_trace(intervals_data, annotations, metadata):
    all_intervals = []
    others_interval = set()
    for interval in intervals_data:
        all_intervals.append(interval)
        if interval["others"] is not None:
            others_interval.update([x["interval_id"] for x in interval["others"]])
    for interval in intervals_data:
        if interval["others"] is not None:
            for o_interval in interval["others"]:
                if o_interval["interval_id"] in others_interval:
                    all_intervals.append(o_interval)
                    others_interval.remove(o_interval["interval_id"])
    all_keys = set([x["interval_id"] for x in all_intervals])
    new_all_intervals = []
    for interval in all_intervals:
        if interval["interval_id"] in all_keys:
            new_all_intervals.append(interval)
            all_keys.remove(interval["interval_id"])
    all_intervals = sorted(new_all_intervals, key=lambda x : int(x["interval_id"].split("|")[2]))
    causal_trace = {interval["interval_id"] : [] for interval in all_intervals}
    for i_idx, interval in enumerate(all_intervals):
        objects = set(get_useful_objects(interval["hoi"]))
        for depend_idx in range(i_idx + 1, len(all_intervals)):
            d_interval = all_intervals[depend_idx]
            d_objects = set(get_useful_objects(d_interval["hoi"]))
            common = objects.intersection(d_objects)
            # Rules for deciding whether this two intervals are directly connected or not
            if len(common) != 0:
                found = False
                # if obj changed in interval and have the before state as d_interval, then they are directly dependent
                for obj in common:
                    if not interval["change"].empty:
                        i_change = interval["change"][interval["change"]["obj"] == obj]
                    else:
                        i_change = pd.DataFrame()
                    if not d_interval["change"].empty:
                        d_change = d_interval["change"][d_interval["change"]["obj"] == obj]
                    else:
                        d_change = pd.DataFrame()
                    # the current action changed status of object
                    if not i_change.empty:
                        i_change_types = set(i_change["change"].values)
                        # 1. if the comparing interval changed the same attribute
                        if not d_change.empty:
                            d_change_types = set(d_change["change"].values)
                            change_common = i_change_types.intersection(d_change_types)
                            if len(change_common) != 0:
                                causal_trace[interval["interval_id"]].append(d_interval["interval_id"])
                                found = True
                                break
                        # 2. if interval changes spatial relationships and the second interval is affected
                        if "spatial relationships" in i_change_types:
                            i_spatial_status = set(i_change[i_change["change"] == "spatial relationships"]["after"].values)
                            if d_interval["relationship"].empty:
                                continue
                            d_spatial_status = d_interval["relationship"].loc[(d_interval["relationship"]["obj"] == obj) & (d_interval["relationship"]["type"] == "before")]
                            if not d_spatial_status.empty:
                                d_spatial_status = set(d_spatial_status["relationship"].values)
                                if len(i_spatial_status.difference(d_spatial_status)) == 0:
                                    causal_trace[interval["interval_id"]].append(d_interval["interval_id"])
                                    found = True
                                    break
                # Unsure whether this two actions are dependent or not
                if not found:
                    causal_trace[interval["interval_id"]].append(f"{d_interval['interval_id']}$unknown")
    for key, vals in causal_trace.items():
        if len(vals) == 0:
            causal_trace[key] = None
        elif len(vals) == 1 and vals[0] == "unknown":
            continue
        else:
            new_vals = []
            for val in vals:
                if val != "unknown":
                    new_vals.append(val)
            causal_trace[key] = new_vals
    return causal_trace


def get_useful_objects(hoi):
    ignoring_objects = ["hand", "table"]
    obj_patterns = "\[[A-Za-z0-9\(\)\-,\ ]*\]"
    objs = re.findall(obj_patterns, hoi)
    objects = [x[1:-1].replace(" ", "").split(",") if "," in x else [x[1: -1]] for x in objs]
    objects = itertools.chain.from_iterable([sorted(x) for x in objects])
    return list(set(objects).difference(set(ignoring_objects)))


def query_person(segments, interval_id):
    """
        Return all relationships of all person in DataFrame,
        Data
    """
    pid = interval_id.split("|")[0].split("@")[1]
    action_anno = []
    found = False
    for segment in segments:
        if segment["obj"] in ["P1", "P2"]:
            if not found:
                for rel_type, rels in segment["rel"].items():
                    for rel in rels:
                        if rel_type == "a_rel":
                            type = "action"
                        elif rel_type == "s_rel":
                            type = "spatial"
                        else:
                            type = "multiagent"
                        rel["type"] = type 
                        rel["relationship"] = "|".join([rel["rel"], "" if "o2" not in rel.keys() else rel["o2"]])
                        rel["relationship"] = qu.pretty_relationship(rel["relationship"], pid)
                        action_anno.append(cp.deepcopy(rel))
            else:
                continue
    action_anno = pd.DataFrame(action_anno)
    action_anno = action_anno.where(pd.notnull(action_anno), None)
    return action_anno


def query_state(segments, interval_id):
    """
            Return all state changes of all objects in DataFrame,
            DataFrame should have columns "obj", "type" (indicating before, after), all state keys
    """
    pid = interval_id.split("|")[0].split("@")[1]
    all_states = []
    all_objs = [segment["obj"] for segment in segments]
    all_objs = list(set(all_objs))
    for obj in all_objs:
        if obj not in ["P1", "P2"]:
            before_state = []
            after_state = []
            for segment in segments:
                if segment["obj"] == obj and segment["type"] == "before":
                    if len(before_state) == 0:
                        state = cp.deepcopy(segment["state"])
                        state["type"] = "before"
                        state["obj"] = qu.pretty_obj(obj, pid)
                        before_state.append(state)

                if segment["obj"] == obj and segment["type"] == "after":
                    if len(after_state) == 0:
                        state = cp.deepcopy(segment["state"])
                        state["type"] = "after"
                        state["obj"] = qu.pretty_obj(obj, pid)
                        after_state.append(state)
            all_states += before_state + after_state
    all_states = pd.DataFrame(all_states)
    return all_states


def query_relation(segments, interval_id):
    """
        Return all relationships of all objects in DataFrame,
        Assume that all relationships of an object has been converted into object-centric description
        DataFrame should have columns "obj", "rel", "r_obj", "relationship", "type"
    """
    pid = interval_id.split("|")[0].split("@")[1]
    relation_anno = []
    all_objs = [segment["obj"] for segment in segments]
    all_objs = list(set(all_objs))
    for obj in all_objs:
        if obj not in ["P1", "P2"]:
            before_state = []
            after_state = []
            for segment in segments:
                if segment["obj"] == obj and segment["type"] == "before":
                    if len(before_state) == 0:
                        for rel in segment["rel"]["s_rel"]:
                            before_state.append({
                                "obj": qu.pretty_obj(rel["o1"], pid), "type": "before",
                                "rel": rel["rel"], "r_obj": rel["o2"],
                                "relationship": qu.pretty_relationship(
                                    rel["rel"] + "|" + rel["o2"], pid
                                )
                            })
                if segment["obj"] == obj and segment["type"] == "after":
                    if len(after_state) == 0:
                        for rel in segment["rel"]["s_rel"]:
                            after_state.append({
                                "obj": rel["o1"], "type": "after",
                                "rel": rel["rel"], "r_obj": rel["o2"],
                                "relationship": qu.pretty_relationship(
                                    rel["rel"] + "|" + rel["o2"], pid
                                )
                            })

            relation_anno += before_state + after_state
    all_relationships = pd.DataFrame(relation_anno)
    return all_relationships


def translate_change(before, after, interval_id):
    pid = interval_id.split("|")[0].split("@")[1]
    changes = {}
    before_rels = sorted(before["rel"]["s_rel"], key=lambda x : ",".join([x["o1"], x["rel"], x["o2"]]))
    after_rels = sorted(after["rel"]["s_rel"], key=lambda x : ",".join([x["o1"], x["rel"], x["o2"]]))
    before_srels = ["|".join([x["rel"], x["o2"]]) for x in before_rels]
    after_srels = ["|".join([x["rel"], x["o2"]]) for x in after_rels]
    before_srel = "$".join([x for x in list(set(before_srels).difference(after_srels))])
    after_srel = "$".join([x for x in list(set(after_srels).difference(before_srels))])

    if len(before_srel) != 0 or len(after_srel) != 0:
        changes["spatial relationships"] = {"before": qu.pretty_relationship(before_srel, pid), "after": qu.pretty_relationship(after_srel, pid)}
    for key, value in before["state"].items():
        if value != after["state"][key]:
            changes[key] = {"before": value, "after": after["state"][key]}
    return changes


def query_change(segments, interval_id):
    """
        Return all state changes of all objects in DataFrame,
        DataFrame should have columns "obj", "change", "before", "after"
    """
    pid = interval_id.split("|")[0].split("@")[1]
    all_changes = []
    changed_objs = set()
    all_objs = [segment["obj"] for segment in segments]
    objs = set(all_objs).difference(["P1", "P2"])
    for obj in objs:
        before_state = None
        after_state = None
        changed_objs.add(obj)
        for segment in segments:
            if segment["obj"] == obj and segment["type"] == "before":
                if before_state is None:
                    before_state = segment

            if segment["obj"] == obj and segment["type"] == "after":
                if after_state is None:
                    after_state = segment

        if before_state is not None and after_state is not None:
            changes = translate_change(before_state, after_state, interval_id)
            for fluent, fluents in changes.items():
                change_dict = {}
                change_dict["obj"] = qu.pretty_obj(obj, pid)
                change_dict["change"] = fluent
                if fluent == "spatial relationships":
                    change_dict["before"] = fluents["before"]
                    change_dict["after"] = fluents["after"]
                else:
                    change_dict["before"] = fluents["before"]
                    change_dict["after"] = fluents["after"]
                all_changes.append(change_dict)

    all_changes = pd.DataFrame(all_changes)
    all_changes = all_changes.where(pd.notnull(all_changes), None)
    return all_changes