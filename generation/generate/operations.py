"""

    Created on 2022/5/16

    @author: Baoxiong Jia

"""

import copy as cp
import itertools
import re
import pandas as pd


def get_useful_objects(hoi):
    ignoring_objects = ["hand", "table"]
    obj_patterns = "\[[A-Za-z0-9\(\)\-,\ ]*\]"
    objs = re.findall(obj_patterns, hoi)
    objects = [x[1:-1].replace(" ", "").split(",") if "," in x else [x[1: -1]] for x in objs]
    objects = itertools.chain.from_iterable([sorted(x) for x in objects])
    return list(set(objects).difference(set(ignoring_objects)))


def select(query, data):
    cp_query = cp.deepcopy(query)
    result = cp.deepcopy(data)
    while len(cp_query) != 0:
        if isinstance(result, dict):
            key = cp_query.pop(0)
            if "@" in key:
                key_type, key_val = key.split("@")
            else:
                key_type = key
                key_val = None
            if key_type in result.keys():
                if key_val is not None:
                    if result[key_type] != key_val:
                        return None
                else:
                    result = result[key_type]
            else:
                return None
        elif isinstance(result, pd.DataFrame):
            if result.empty:
                return None
            keys = cp_query.pop(0).split("@")
            if len(keys) == 1:
                if len(keys[0].split("$")) == 2:
                    query_col, query_val = keys[0].split("$")
                else:
                    query_col = keys[0].split("$")[0]
                    query_val = ""
                if query_col not in result:
                    return None
                result = list(result[query_col].values)
                if query_val != "":
                    while not isinstance(result, str):
                        if len(result) != 1:
                            return None
                        result = result[0]
                    if result != query_val:
                        return None
            else:
                key_type, key = keys
                if key_type not in result:
                    return None
                result = result.loc[result[key_type] == key]
                if result.empty:
                    return None
        elif result is None:
            return None
    return result


def filter(filter_keys, current_intervals, intervals_data, causal_trace):
    """
    :param filter_key:          sequence of read out frames [hoi], [task], [relationships]
    :param current_intervals:   ids of current intervals in the global array
    :param intervals_data:      global array
    :return:
    """
    if isinstance(current_intervals, str):
        new_current_intervals = []
        for (idx, x) in enumerate(intervals_data):
            if current_intervals == "video":
                if not x["is_pred"]:
                    new_current_intervals.append((idx, x))
            else:
                new_current_intervals.append((idx, x))
        current_intervals = new_current_intervals
    if len(filter_keys) == 0:
        return current_intervals
    new_current_intervals = []
    for comp_interval in current_intervals:
        aid, interval = comp_interval
        ori_interval = cp.deepcopy(interval)
        result = select(filter_keys, interval)
        if result is not None and len(result) != 0:
            if "others" not in filter_keys:
                new_current_intervals.append((aid, ori_interval))
            else:
                new_current_intervals += [(o_idx, o_interval) for o_idx, o_interval in enumerate(result)]
    if len(new_current_intervals) == 0:
        return None
    refined_keys = set([interval[1]["interval_id"] for interval in new_current_intervals])
    refined_current_intervals = []
    for new_interval in new_current_intervals:
        if new_interval[1]["interval_id"] in refined_keys:
            refined_keys.remove(new_interval[1]["interval_id"])
            refined_current_intervals.append(new_interval)
    return refined_current_intervals


def iterate_until(direction, current_intervals, intervals_data, causal_trace):
    """
    :param direction:           forward corresponding to first and backward corresponding to last
    :param current_intervals:
    :return:                    reduced list with only one item
    """
    if len(current_intervals) == 0:
        return None
    if direction == "forward":
        return current_intervals[0]
    elif direction == "backward":
        return current_intervals[-1]
    else:
        raise NotImplementedError(f"Direction {direction} is not specified for IterateUntil")


def localize(time_hint, current_interval, all_intervals, intervals_data, causal_trace):
    """
    :param time_hint:   before, after or while
    :param key: a unique action_id
    :param intervals_data:
    :return: all interval data before after localizing the interval
    """
    new_intervals_data = []
    action_id, interval = current_interval
    if time_hint == "is":
        new_intervals_data.append(current_interval)
    elif time_hint == "while":
        # parallel action
        _, hoi, start, end = all_intervals[action_id][1]["interval_id"].split("|")
        for aid, interval in all_intervals:
            if aid == action_id:
                continue
            _, i_hoi, i_start, i_end = interval["interval_id"].split("|")
            if (int(i_start) < int(end) and int(i_start) >= int(start)) or \
                    (int(i_end) < int(end) and int(i_end) >= int(start)) and not interval["is_pred"]:
                new_intervals_data.append((aid, interval))
    elif time_hint == "before":
        # before actions
        vid, hoi, start, end = all_intervals[action_id][1]["interval_id"].split("|")
        for aid, interval in all_intervals:
            if aid == action_id:
                continue
            _, i_hoi, i_start, i_end = interval["interval_id"].split("|")
            if int(i_end) < int(start) and not interval["is_pred"]:
                new_intervals_data.append((aid, interval))
    elif time_hint == "after":
        vid, hoi, start, end = all_intervals[action_id][1]["interval_id"].split("|")
        for aid, interval in all_intervals:
            if aid == action_id:
                continue
            _, i_hoi, i_start, i_end = interval["interval_id"].split("|")
            if int(i_start) > int(end) and not interval["is_pred"]:
                new_intervals_data.append((aid, interval))
    else:
        raise NotImplementedError(f"Unclear time hint {time_hint}")
    if len(new_intervals_data) == 0:
        new_intervals_data = None
    return new_intervals_data


def query(query, current_interval, intervals_data, causal_trace):
    result = select(query, current_interval[1])
    if result is None:
        return None
    while not isinstance(result, str):
        if len(result) != 1:
            return None
        result = result[0]
    if "visible" in result:
        if "invisible" in result:
            return "no"
        else:
            return "yes"
    return result


def verify(query, current_interval, intervals_data, causal_trace):
    aid, result = current_interval
    result = select(query, result)
    if result is not None:
        return "yes"
    else:
        return "no"


def pred(current_intervals, intervals_data, causal_trace):
    """
    Prediction of future
    :param intervals_data:
    :return:
    """
    if isinstance(current_intervals, str):
        new_current_intervals = []
        for (idx, x) in enumerate(intervals_data):
            new_current_intervals.append((idx, x))
        current_intervals = new_current_intervals
    new_current_intervals = []
    for cid, interval in current_intervals:
        if interval["is_pred"]:
            new_current_intervals.append((cid, interval))
    if len(new_current_intervals) == 0:
        return None
    return new_current_intervals


def only(current_intervals, intervals_data, causal_trace):
    """
    :param intervals_data:  should be a one element interval
    :return:
    """
    if len(current_intervals) == 1:
        return current_intervals[0]
    else:
        return None


def counterfactual(query, current_intervals, intervals_data, causal_trace):
    """
    Counterfactual of results
    """
    def dfs(interval_id, causal_trace, interval_objs, affecting, intervals_data):
        # if interval_id == "30k-22-1-1@P2|Put [juicer] to [juicer-base] using [hand]|2723|2839":
        #     print("debugging")
        unknown = False
        if "unknown" in interval_id:
            interval_id = interval_id.split("$")[0]
            unknown = True
        affecting_ids = causal_trace[interval_id]
        if affecting_ids is None:
            return
        for affecting_id in affecting_ids:
            inner_unknown = False
            if "unknown" in affecting_id:
                inner_unknown = True
            affecting_objs = get_useful_objects(affecting_id.split("|")[1])
            if len(set(interval_objs).intersection(affecting_objs)) == 0:
                continue
            if unknown or inner_unknown:
                if "unknown" not in affecting_id:
                    affecting_id = affecting_id + "$unknown"
            affecting.append(affecting_id)
            dfs(affecting_id, causal_trace, interval_objs, affecting, intervals_data)
            inner_unknown = False
        return

    if isinstance(current_intervals, str):
        new_current_intervals = []
        for (idx, x) in enumerate(intervals_data):
            if current_intervals == "video":
                if not x["is_pred"]:
                    new_current_intervals.append((idx, x))
            else:
                new_current_intervals.append((idx, x))
        current_intervals = new_current_intervals
    if "others" not in query:
        intervals = filter(query, current_intervals, intervals_data, causal_trace)
        if intervals is None:
            return None
        counterfactual_interval = only(intervals, intervals_data, causal_trace)
    else:
        query = query[1:]
        others_intervals = filter(["others"], "video", intervals_data, causal_trace)
        if others_intervals is None:
            return None
        filter_others_intervals = filter(query, others_intervals, intervals_data, causal_trace)
        if filter_others_intervals is None:
            return None
        counterfactual_interval = only(filter_others_intervals, intervals_data, causal_trace)
    if counterfactual_interval is not None:
        interval_id = counterfactual_interval[1]["interval_id"]
        related_ids = []
        interval_objs = get_useful_objects(interval_id.split("|")[1])
        dfs(interval_id, causal_trace, interval_objs, related_ids, intervals_data)
        related_ids = list(set(related_ids))
        all_after_intervals = []
        all_after_interval_ids = []
        after = False
        for interval in current_intervals:
            if after:
                all_after_intervals.append(cp.deepcopy(interval))
                all_after_interval_ids.append(interval[1]["interval_id"])
            if interval[1]["interval_id"] == interval_id or \
                int(interval[1]["interval_id"].split("|")[2]) > int(interval_id.split("|")[2]):
                after = True
        affecting = set()
        for related_id in related_ids:
            if "unknown" not in related_id:
                affecting.add(related_id)
        unaffecting = set(all_after_interval_ids).difference([x.replace("$unknown", "") for x in related_ids])
        for interval in all_after_intervals:
            if interval[1]["interval_id"] in affecting:
                interval[1]["executable"] = "no"
            elif interval[1]["interval_id"] in unaffecting:
                continue
            else:
                interval[1]["executable"] = "unknown"
        return all_after_intervals
    else:
        return None


def depend(intervals1, intervals2, intervals_data, causal_trace):
    interval_id1 = intervals1[1]["interval_id"]
    interval_id2 = intervals2[1]["interval_id"]
    def dfs(id1, id2, causal_trace):
        unknown = False
        if "unknown" in id1:
            unknown = True
            id1 = id1.split("$")[0]
        affecting_ids = causal_trace[id1]
        if affecting_ids is not None:
            for affecting_id in affecting_ids:
                inner_unknown = False
                if "unknown" in affecting_id:
                    inner_unknown = True
                if unknown or inner_unknown:
                    if "unknown" not in affecting_id:
                        affecting_id += "$unknown"
                if id2 in affecting_id:
                    if "unknown" in affecting_id:
                        return "unknown"
                    else:
                        return "yes"
                else:
                    dfs(affecting_id, id2, causal_trace)
        else:
            return "no"
    related = dfs(interval_id1, interval_id2, causal_trace)
    return related                


if __name__ == "__main__":
    pass