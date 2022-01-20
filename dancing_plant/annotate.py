
"""
Produces video of trace points over time annotated
on the original frames.
"""


import os
import os.path as osp
import cv2
import glob
import numpy as np

from tqdm import tqdm
from natsort import natsorted, ns

import dancing_plant
from dancing_plant.track import rp, out_pfx as track_out_pfx
from dancing_plant.cluster import get_num_part

IMAGE_PATH = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed"
TRACE_PATH = "/home/lowell/dancing-plant/RAFT/dtw_corr_traces"  # assumes traces in format X_trace{i}.csv and Y_trace{i}.csv where 'i' is partition
TRACE_SFX = ""  # can insert suffix after expected format above (leave as empty str or None for no suffix)
SAVE_PATH = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed/annotated-dtw-corr"

NUM_PART = 2
SAMPLE_FREQ = 1
CIRC_SIZE = 6


def annotate(image_path, trace_path, num_part, sample_freq, circ_size, save_path=None, trace_sfx=""):
    if trace_sfx is None:
        trace_sfx = ""

    # load trace
    if num_part is None or num_part <= 1:
        x_split_paths = [osp.join(trace_path, f"X_trace{trace_sfx}.csv")]
        y_split_paths = [osp.join(trace_path, f"Y_trace{trace_sfx}.csv")]
    else:
        x_split_paths = [osp.join(trace_path, f"X_trace{idx}{trace_sfx}.csv") for idx in range(num_part)]
        y_split_paths = [osp.join(trace_path, f"Y_trace{idx}{trace_sfx}.csv") for idx in range(num_part)]

    x_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in x_split_paths]
    y_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in y_split_paths]

    # get distinct color for each trace
    tcolor = [dict() for _ in range(num_part)]
    for split_idx in range(num_part):
        for i in range(len(x_split_traces[split_idx])):  # y_split_traces should be same length
            b, g, r = rp(), rp(), rp()
            tcolor[split_idx][i] = (b, g, r)

    # load, annotate, and save each image
    image_paths = glob.glob(osp.join(image_path, "*.jpg")) + \
                  glob.glob(osp.join(image_path, "*.png"))
    image_paths = natsorted(image_paths)

    if save_path is None:
        save_path = osp.join(image_path, "annotated")
    
    if not osp.exists(save_path):
        os.makedirs(save_path)

    sampled_image_paths = list(filter(lambda x: x[0] % sample_freq == 0, enumerate(image_paths)))
    sampled_image_paths = [tup[1] for tup in sampled_image_paths]  # remove uncontiguous enumeration

    print(f"Annotating {len(sampled_image_paths)} images from {image_path}; sample frequency = {sample_freq}")
    for img_idx, ipath in tqdm(enumerate(sampled_image_paths)):
        img = cv2.imread(ipath)

        print(f"img_idx={img_idx} image path={ipath}")

        for split_idx in range(num_part):
            xtr = x_split_traces[split_idx]
            ytr = y_split_traces[split_idx]
            tcol = tcolor[split_idx]

            for trace_idx in range(len(xtr)):
                if img_idx==229:
                    print(f"trace_idx={trace_idx}  img_idx={img_idx}") 
                    print(xtr)               
                x = xtr[trace_idx, img_idx]
                y = ytr[trace_idx, img_idx]
                cv2.circle(img, (x, y), circ_size, tcol[trace_idx], -1)

        fout_path = osp.join(save_path, osp.basename(ipath))
        cv2.imwrite(fout_path, img)


def create_video(frame_path, fps):
    import imageio
    frame_names = glob.glob(osp.join(frame_path, "*.jpg")) + \
                  glob.glob(osp.join(frame_path, "*.png"))
    sorted_names = natsorted(frame_names)
    frames = [cv2.cvtColor(cv2.imread(sn), cv2.COLOR_BGR2RGB) for sn in sorted_names]
    vid_path = osp.join(frame_path, "annotations.mp4")
    imageio.mimwrite(vid_path, frames, fps=fps)


def run_annotate_with_defaults(image_path, trace_path, sample_freq, circ_size, fps=25):
    """Also automatically creates a video with annotated frames."""
    ## trace_path = track_out_pfx("")

    if not os.listdir(trace_path):
        raise EnvironmentError("No trace CSVs found at expected trace path: "
            f"{trace_path}\nRun gen_flow.py and gen_trace.py first.")

    num_part = get_num_part(trace_path)

    save_path = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "annotated")
    if not osp.exists(save_path):
        os.makedirs(save_path)

    annotate(
        image_path,
        trace_path,
        num_part,
        sample_freq,
        circ_size,
        save_path
    )

    create_video(save_path, fps)



if __name__ == "__main__":
    annotate(
        IMAGE_PATH, 
        TRACE_PATH, 
        NUM_PART,
        SAMPLE_FREQ,
        CIRC_SIZE,
        SAVE_PATH,
        TRACE_SFX
    )
