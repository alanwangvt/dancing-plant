
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

from track import rp

image_path = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed"
trace_path = "/home/lowell/dancing-plant/RAFT/dtw_corr_traces"  # assumes traces in format X_trace{i}.csv and Y_trace{i}.csv where 'i' is partition
trace_sfx = ""  # can insert suffix after expected format above (leave as empty str for no suffix)
annotation_title = "annotated-dtw-corr"

NUM_PART = 2
CIRC_SIZE = 6


if __name__ == "__main__":
    
    # load trace
    if NUM_PART is None or NUM_PART <= 1:
        x_split_paths = [osp.join(trace_path, f"X_trace{trace_sfx}.csv")]
        y_split_paths = [osp.join(trace_path, f"Y_trace{trace_sfx}.csv")]
    else:
        x_split_paths = [osp.join(trace_path, f"X_trace{idx}{trace_sfx}.csv") for idx in range(NUM_PART)]
        y_split_paths = [osp.join(trace_path, f"Y_trace{idx}{trace_sfx}.csv") for idx in range(NUM_PART)]

    x_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in x_split_paths]
    y_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in y_split_paths]

    # get distinct color for each trace
    tcolor = [dict() for _ in range(NUM_PART)]
    for split_idx in range(NUM_PART):
        for i in range(len(x_split_traces[split_idx])):  # y_split_traces should be same length
            b, g, r = rp(), rp(), rp()
            tcolor[split_idx][i] = (b, g, r)

    # load, annotate, and save each image
    image_paths = glob.glob(osp.join(image_path, "*.jpg")) + \
                  glob.glob(osp.join(image_path, "*.png"))
    image_paths = natsorted(image_paths)

    save_path = osp.join(image_path, annotation_title)
    if not osp.exists(save_path):
        os.makedirs(save_path)

    print(f"Annotating {len(image_paths)} images from {image_path}")
    for img_idx, ipath in tqdm(enumerate(image_paths)):
        img = cv2.imread(image_paths[img_idx])

        for split_idx in range(NUM_PART):
            xtr = x_split_traces[split_idx]
            ytr = y_split_traces[split_idx]
            tcol = tcolor[split_idx]

            for trace_idx in range(len(xtr)):
                x = xtr[trace_idx, img_idx]
                y = ytr[trace_idx, img_idx]
                cv2.circle(img, (x, y), CIRC_SIZE, tcol[trace_idx], -1)

        fout_path = osp.join(save_path, osp.basename(ipath))
        cv2.imwrite(fout_path, img)