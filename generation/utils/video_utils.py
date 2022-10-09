"""

    Created on 2022/5/12

    @author: Baoxiong Jia

"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils.log_utils import LOGGER
from utils.anno_utils import parse_interval


def generate_intervals(metadata, prediction_window_length=3, fps=24, min_length=20):
    clips = []
    avg_num_intervals = []
    avg_duration = []
    intervals = metadata["intervals"]
    for vid, v_intervals in intervals.items():
        if vid == "30l-12-2-1":
            continue
        for pid, p_intervals in v_intervals.items():
            vid_length = metadata["lengths"][vid][pid]
            clip_num = int(vid_length / (fps * min_length))
            occupied = [False for _ in range(len(p_intervals))]
            starts = [int(x) for x in np.linspace(0, vid_length - 1, clip_num)[:-1]]
            ends = [int(x) for x in np.linspace(0, vid_length - 1, clip_num)[1:]]
            for start, end in zip(starts, ends):
                current_clips = []
                actual_start = start
                actual_end = end
                end_idx = -1
                for i_idx, interval in enumerate(p_intervals):
                    if occupied[i_idx]:
                        continue
                    interval = parse_interval(interval)
                    if interval["start"] >= start and interval["end"] < end:
                        # this action is within the time interval
                        current_clips.append(i_idx)
                        occupied[i_idx] = True
                        end_idx = i_idx
                    elif interval["start"] < start and interval["end"] >= start and interval["end"] < end:
                        # this interval starts in the middle of an action
                        actual_start = interval["start"]
                        current_clips.append(i_idx)
                        occupied[i_idx] = True
                        end_idx = i_idx
                    elif interval["start"] >= start and interval["start"] < end and interval["end"] >= end:
                        # this interval ends in the middle of an action
                        actual_end = interval["end"]
                        current_clips.append(i_idx)
                        occupied[i_idx] = True
                        end_idx = i_idx
                    else:
                        break
                if end_idx == -1:
                    pass
                else:
                    predict_clips =[x for x in range(end_idx + 1, min(end_idx + 1 + prediction_window_length, len(p_intervals)))]
                    pred_start = parse_interval(p_intervals[predict_clips[0]])["start"] if len(predict_clips) > 0 else actual_end
                    actual_end = parse_interval(p_intervals[predict_clips[-1]])["end"] if len(predict_clips) > 0 else actual_end
                    clip_id = f"{vid}|{pid}|{actual_start}|{actual_end}"
                    action_ids = current_clips + predict_clips
                    action_ids = f"{vid}|{pid}|{action_ids[0]}-{action_ids[-1]}"
                    clips.append((clip_id, action_ids, pred_start))
                    avg_num_intervals.append(len(current_clips))
                    avg_duration.append((pred_start - actual_start + 1) / fps)
    LOGGER.info(f"Total number of clips available with time {min_length}s "
                f"is {len(clips)} and contains {sum(avg_num_intervals) / len(avg_num_intervals)}"
                f" intervals on average with (min {min(avg_num_intervals)} and max {max(avg_num_intervals)}) and "
                f"average duration {sum(avg_duration) / len(avg_duration)}.")
    return clips


def resize_all_imgs(data_path, short_scale=256):
    for video in tqdm([x for x in Path(data_path).iterdir()]):
        for directory in video.iterdir():
            if directory.is_file():
                continue
            for image_path in directory.iterdir():
                if str(image_path.stem).startswith("img"):
                    image = cv2.imread(str(image_path))
                    if "ori" not in str(image_path):
                        height, width = image.shape[0], image.shape[1]
                        if (width != 341 and width != 455) or height != 256:
                            tqdm.write(f"Resizing {str(image_path)}")
                            resized_image = cv2.resize(image, dsize=(341, 256), interpolation=cv2.INTER_LINEAR)
                            cv2.imwrite(str(image_path), resized_image)
                            cv2.imwrite(str(image_path).replace("_", "_ori_"), image)


if __name__ == "__main__":
    pass

