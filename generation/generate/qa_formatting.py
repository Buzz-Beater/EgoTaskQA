"""

    Created on 2022/5/16

    @author: Baoxiong Jia
"""

import re
import copy as cp
from generate.executor import execute


def pretty_fluent(pid, fluents):
    to_P1 = "me" if pid == "P1" else "the other person"
    to_P2 = "me" if pid == "P2" else "the other person"
    pretty_fluent_value_dict = {
        "visible": {"yes" : "visible", "no": "occluded", "N/A": "unknown"},
        "visible_to_P1": {"yes" : f"visible to {to_P1}", "no": f"invisible to {to_P1}", "N/A": "unknown"},
        "visible_to_P2": {"yes" : f"visible to {to_P2}", "no": f"invisible to {to_P2}", "N/A": "unknown"},
        "edible": {"yes": "edible", "no": "can not be eaten", "N/A": "unknown"},
        "cuttable": {"yes": "cuttable", "no": "not cuttable", "N/A": "unknown"},
        "openable": {"yes": "openable", "no": "can not be opened", "N/A": "unknown"},
        "turnable": {"yes": "could be turned on", "no": "can not be turned on", "N/A": "unknown"},
        "boiled": {"yes": "boiled", "no": "in room temperature", "N/A": "unknown"},
        "cooked": {"yes": "cooked", "no": "raw", "N/A": "unknown"},
        "on": {"yes": "on", "no": "off", "N/A": "unknown"},
        "opened": {"yes": "opened", "no": "closed", "N/A": "unknown"},
        "wrapped": {"yes": "wrapped", "no": "unwrapped", "N/A": "unknown"},
        "emptiness": {"yes": "empty", "no": "nonempty", "N/A": "unknown"},
        "mixing": {"yes": "mixing", "no": "not mixing", "N/A": "unknown"},
        "cleanliness": {"yes": "clean", "no": "dirty", "N/A": "unknown"},
        "shape": {"whole": "a whole", "part": "in part", "diced": "diced", "fluid": "in fluid", "N/A": "unknown"}
    }
    pretty_fluent_name = {
        "visible": "visibility",
        "visible_to_P1": f"visibility to {to_P1}",
        "visible_to_P2": f"visibility to {to_P2}",
        "edible": "edibility",
        "cuttable": "cuttability",
        "openable": "openability",
        "turnable": "switchability",
        "boiled": "temperature",
        "cooked": "cookedness",
        "on": "poweredness",
        "opened": "openess",
        "wrapped": "wrappedness",
        "emptiness": "emptiness",
        "mixing": "state of mixture",
        "cleanliness": "cleanliness",
        "shape": "shape",
        "spatial": "spatial relationships"
    }
    new_fluents = {}
    for fluent_key, fluent_val in fluents.items():
        if fluent_val == "unknow":
            new_fluents[pretty_fluent_name[fluent_key]] = "unknown"
        else:
            new_fluents[pretty_fluent_name[fluent_key]] = pretty_fluent_value_dict[fluent_key][fluent_val]
    return new_fluents


def pretty_fluent_name(fluent, pid):
    to_P1 = "me" if pid == "P1" else "the other person"
    to_P2 = "me" if pid == "P2" else "the other person"
    pretty_fluent_name = {
        "visible": "visibility",
        "visible_to_P1": f"visibility to {to_P1}",
        "visible_to_P2": f"visibility to {to_P2}",
        "edible": "edibility",
        "cuttable": "cuttability",
        "openable": "openability",
        "turnable": "switchability",
        "boiled": "temperature",
        "cooked": "cookedness",
        "on": "poweredness",
        "opened": "openess",
        "wrapped": "wrappedness",
        "emptiness": "emptiness",
        "mixing": "state of mixture",
        "cleanliness": "cleanliness",
        "shape": "shape",
        "spatial": "spatial relationships"
    }
    return pretty_fluent_name[fluent]


def pretty_fluent_vals(pid="P1"):
    to_P1 = "me" if pid == "P1" else "the other person"
    to_P2 = "me" if pid == "P2" else "the other person"
    pretty_fluent_value_dict = {
        "visible": {"yes": "visible", "no": "occluded", "N/A": "unknown"},
        "visible_to_P1": {"yes": f"visible to {to_P1}", "no": f"invisible to {to_P1}", "N/A": "unknown"},
        "visible_to_P2": {"yes": f"visible to {to_P2}", "no": f"invisible to {to_P2}", "N/A": "unknown"},
        "edible": {"yes": "edible", "no": "can not be eaten", "N/A": "unknown"},
        "cuttable": {"yes": "cuttable", "no": "not cuttable", "N/A": "unknown"},
        "openable": {"yes": "openable", "no": "can not be opened", "N/A": "unknown"},
        "turnable": {"yes": "could be turned on", "no": "can not be turned on", "N/A": "unknown"},
        "boiled": {"yes": "boiled", "no": "in room temperature", "N/A": "unknown"},
        "cooked": {"yes": "cooked", "no": "raw", "N/A": "unknown"},
        "on": {"yes": "on", "no": "off", "N/A": "unknown"},
        "opened": {"yes": "opened", "no": "closed", "N/A": "unknown"},
        "wrapped": {"yes": "wrapped", "no": "unwrapped", "N/A": "unknown"},
        "emptiness": {"yes": "empty", "no": "nonempty", "N/A": "unknown"},
        "mixing": {"yes": "mixing", "no": "not mixing", "N/A": "unknown"},
        "cleanliness": {"yes": "clean", "no": "dirty", "N/A": "unknown"},
        "shape": {"whole": "a whole", "part": "in part", "diced": "diced", "fluid": "in fluid", "N/A": "unknown"}
    }
    all_value_set = set()
    for key, dct in pretty_fluent_value_dict.items():
        for ans, pretty_ans in dct.items():
            all_value_set.add(pretty_ans)
    return list(all_value_set)


def pretty_hoi(hoi, pid):
    new_hoi = ""
    within_paraenthesis = False
    within_bracket = False
    for ch in hoi:
        if within_paraenthesis:
            continue
        if within_bracket and ch == " ":
            continue
        if ch == "[":
            within_bracket = True
            new_hoi += ""
        elif ch == "]":
            within_bracket = False
            new_hoi += ""
        elif ch == "-":
            new_hoi += "-"
        elif ch == "(":
            within_paraenthesis = True
        elif ch == ")":
            within_paraenthesis = False
        elif ch == ",":
            new_hoi += " and "
        else:
            new_hoi += ch
    new_hoi = new_hoi.rstrip()
    if "P1" in new_hoi:
        new_hoi = new_hoi.replace("P1", "me" if "P1" == pid else "the other person")
    if "P2" in new_hoi:
        new_hoi = new_hoi.replace("P2", "me" if "P2" == pid else "the other person")
    if " using hand" in new_hoi:
        new_hoi = new_hoi.replace(" using hand", "")
    hoi = hoi.lower()
    return new_hoi


def anonymize(hoi, pid, display_num=2):
    new_hoi = ""
    within_paraenthesis = False
    within_bracket = False
    num_bracket = 0
    for ch in hoi:
        if within_paraenthesis:
            continue
        if within_bracket and ch == " ":
            continue
        if ch == "[":
            num_bracket += 1
            within_bracket = True
            new_hoi += ""
            continue
        if ch == "]":
            within_bracket = False
            if num_bracket <= display_num:
                new_hoi += "something"   
            continue
        if within_bracket:
            if num_bracket <= display_num:
                continue
        if ch == "(":
            within_paraenthesis = True
            continue
        if ch == ")":
            within_paraenthesis = False
            continue
        if ch == ",":
            new_hoi += " and "
            continue
        new_hoi += ch
    new_hoi = new_hoi.rstrip()
    if "P1" in new_hoi:
        new_hoi = new_hoi.replace("P1", "me" if "P1" == pid else "the other person")
    if "P2" in new_hoi:
        new_hoi = new_hoi.replace("P2", "me" if "P2" == pid else "the other person")
    if " using hand" in new_hoi:
        new_hoi = new_hoi.replace(" using hand", "")
    new_hoi = new_hoi.lower()
    return new_hoi


def pretty_obj(obj, pid):
    new_obj = ""
    within_paraenthesis = False
    for ch in obj:
        if within_paraenthesis:
            continue
        if ch == "-":
            new_obj += "-"
        elif ch == "(":
            within_paraenthesis = True
        elif ch == ")":
            within_paraenthesis = False
        else:
            new_obj += ch
    if "P1" in new_obj:
        new_obj = new_obj.replace("P1", "me" if "P1" == pid else "the other person")
    if "P2" in new_obj:
        new_obj = new_obj.replace("P2", "me" if "P2" == pid else "the other person")
    new_obj = new_obj.lower()
    return new_obj


def pretty_relationship(relationships, pid):
    if isinstance(relationships, str):
        relationships = relationships.split("$")
    relationships = [" ".join(x.split("|")).replace(">", "").replace("<", "").replace("-", " ") for x in relationships]
    relationships = " and ".join(relationships)
    if "P1" in relationships:
        relationships = relationships.replace("P1", "me" if "P1" == pid else "the other person")
    if "P2" in relationships:
        relationships = relationships.replace("P2", "me" if "P2" == pid else "the other person")
    relationships = relationships.lower()
    return relationships


def compress_anno(anno):
    new_anno = dict()
    pid = anno["pid"]
    for key, items in anno.items():
        if key in ["insert", "swap", "pid", "a_idx", "anno_id"]:
            continue
        elif key == "rel":
            new_rels = {rel_type : [] for rel_type in items.keys()}
            for rel_type, rels in items.items():
                for rel in rels:
                    if rel["o1"] in ["P1", "P2"]:
                        rel["o1"] = "me" if pid == rel["o1"] else "the other person"
                    new_rel = {"o1" : rel["o1"], "rel": rel["rel"]}
                    if "o2" in rel.keys():
                        if rel["o2"] in ["P1", "P2"]:
                            rel["o2"] = "me" if pid == rel["o2"] else "the other person"
                        new_rel["o2"] = rel["o2"]
                    new_rels[rel_type].append(new_rel)
            new_anno[key] = new_rels
        elif key == "state":
            new_fluents = {}
            for fluent, fluent_val in items.items():
                if fluent_val in ["N/A", "unknow"]:
                    new_fluents[fluent] = "N/A"
                else:
                    new_fluents[fluent] = fluent_val
            new_fluents = pretty_fluent(pid, new_fluents)
            new_anno[key] = new_fluents
        else:
            new_anno[key] = items
    return new_anno


def post_process(result):
    if result["answer"] in result["question"] and \
        result["answer"] not in ["yes", "no", "unknown"] and \
        result["semantic"] == "object":
        return None
    if result["answer"] == "unknown":
        return None
    if result["answer"] == "emptiness":
        # for plants, emptiness means wateredness
        if "query([change, obj@'plant', change$]" in result["program"]:
            result["answer"] = "wateredness"
            return result
        return result
    # Remove visibility to me qas
    if result["answer"] in ["invisible to the other person", "visible to the other person"] \
        and result["interval"][4] != "2":
        return None
    # Remove shape being a whole qas
    if result["answer"] in ["a whole"]:
        return None
    if "do not stand-up" in result["question"]:
        return None
    if result["semantic"] == "task":
        if "," in result["answer"]:
            return None
    return result


def process_obj_vis(qa, interval_data, causal_trace, vis_num):
    """
    Control the number of objects visible in questions
    """
    def dfs(program, act2hoi, answer, interval_data, causal_trace, current_map):
        action_pattern = "action@'[\{\}a-z0-9 \-]*'"
        all_act_query = re.findall(action_pattern, program)
        results = []
        if len(all_act_query) == 0:
            if execute(program, interval_data, causal_trace) == answer:
                return [cp.deepcopy(current_map)]
            else:
                return []
        act_query = all_act_query[0]
        action = act_query.split("@")[1][1:-1]
        all_potential_hois = act2hoi[action]
        for comp_hoi in all_potential_hois:
            hoi, cur_pid = comp_hoi.split("$")
            substitute = f"hoi@'{pretty_hoi(hoi, cur_pid)}'"
            new_current_map = cp.deepcopy(current_map)
            new_current_map[action] = hoi + "$" + cur_pid
            new_program = cp.deepcopy(program)
            new_program = new_program.replace(act_query, substitute)
            result = dfs(new_program, act2hoi, answer, interval_data, causal_trace, new_current_map)
            results += result
        return results

    if vis_num == 2:
        return [qa]
    all_hois = {}
    pid = qa["interval"].split("|")[1]
    for interval in interval_data:
        all_hois[interval["ori_hoi"] + "$" + pid] = interval["action"]
        if interval["others"] is not None:
            o_pid = "P2" if pid == "P1" else "P1"
            for o_interval in interval["others"]:
                all_hois[o_interval["ori_hoi"] + "$" + o_pid] = o_interval["action"]
    act2hoi = {}
    for hoi, act in all_hois.items():
        if act not in act2hoi.keys():
            act2hoi[act] = [hoi]
        else:
            act2hoi[act].append(hoi)
    valid_substitution = dfs(qa["program"], act2hoi, qa["answer"], interval_data, causal_trace, {})
    
    if len(valid_substitution) == 0:
        return [qa]
    results = []
    all_valid_questions = set()
    for valid_sub in valid_substitution:
        new_qa = cp.deepcopy(qa)
        for key, val in valid_sub.items():
            cur_hoi, cur_pid = val.split("$")
            cur_hoi = anonymize(cur_hoi, cur_pid, display_num=vis_num)
            new_qa["question"] = new_qa["question"].replace(f"{key}", cur_hoi)
        if new_qa["question"] in all_valid_questions:
            continue
        else:
            all_valid_questions.add(new_qa["question"])
        pattern = "'\{[a-z0-9]*\}:[A-Za-z0-9\- ]*'"
        need_sups = re.findall(pattern, new_qa["question"])
        for need_sup in need_sups:
            new_qa["question"] = new_qa["question"].replace(need_sup, need_sup[1:-1].split(":")[1])
        digits = set()
        for ch_idx, ch in enumerate(new_qa["question"]):
            if ch.isdigit():
                digits.add(ch)
        for ch_idx, ch in enumerate(new_qa["answer"]):
            if ch.isdigit():
                digits.add(ch)
        for digit in digits:
            new_qa["question"] = new_qa["question"].replace(digit, "")
            new_qa["answer"] = new_qa["answer"].replace(digit, "")
        results.append(cp.deepcopy(new_qa))
    if len(results) > 1:
        print(qa)
        print(valid_substitution)
        print("found duplicate")
    return results