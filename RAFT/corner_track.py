import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import json
import time

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm
from natsort import natsorted, ns

from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2hsv

# image_path = "/home/lowell/dancing-plant/DPI/selected pictures from 07-21-2020/WIN_20200721_15_34_14_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/selected pictures from 07-21-2020/raft-flow-raw-5"
# image_path = "/home/lowell/dancing-plant/DPI/07-27-2020/military-time/Webcam Shot Date July 25 2020 Time 23.52.57.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/07-27-2020/military-time/raft-flow-raw-5"
# image_path = "/home/lowell/dancing-plant/DPI/07-31-2020-Azura/WIN_20200731_19_12_08_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/07-31-2020-Azura/raft-flow-raw-20"
# image_path = "/home/lowell/dancing-plant/DPI/07-15-2020/pictures/WIN_20200715_12_37_37_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/07-15-2020/pictures/raft-flow-raw-10"
# image_path = "/home/lowell/dancing-plant/DPI/selected from 07-22-2020/WIN_20200722_17_23_30_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/selected from 07-22-2020/raft-flow-raw-10"
# image_path = "/home/lowell/dancing-plant/DPI/11-01-2020_Azura/military-time/Webcam Shot Date November 2 2020 Time 07.08.02.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/11-01-2020_Azura/military-time/raft-flow-raw-3"
# image_path = "/home/lowell/dancing-plant/DPI/11-03-2020Azura/military-time/Webcam Shot Date November 3 2020 Time 16.23.49.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/11-03-2020Azura/military-time/raft-flow-raw-5"
# image_path = "/mnt/slow_ssd/lowell/DPI/01-24-2021/military-time/Webcam Shot Date January 24 2021 Time 22.27.50.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/01-24-2021/military-time/raft-flow-raw-5"
image_path = "/home/lowell/dancing-plant/DPI/02-01-2021/military-time/Webcam Shot Date February 1 2021 Time 17.25.34.jpg"
flow_path = "/mnt/slow_ssd/lowell/DPI/02-01-2021/military-time/raft-flow-raw-1"

LOAD_STEPS = 4  # Number of process steps to break trace genration into, where only 1 / LOAD_STEPS fraction of flow images will be loaded at a time

X_SPLITS = (950,)  # List of X coordinates to partition the tracks by vertically
SHOW_PARTITION = True  # Will draw vertical lines where the tracks are partitioned

JUST_CORNERS = False  # Will only run corner detection if true, otherwise will run tracking after corner detection

NUM_TRACE = 25  # Top NUM_TRACE most mobile corner traces will be kept

CORNER_THRESH = 0.00025  # R scores must be greater than this faction of the max R score
BOCK_SIZE = 2  # Size of local area used when creating R scores from gradients
SOBEL_SIZE = 9  # Size of sobel kernel used in corner detection
FREE_K = 0.00  # Parameter trading off between edge and corner detection (higher is stricter on corner detections, lower will allow more edges)
NONM_SIZE = 80  # Nonmax suppress tile size
NONM_NUM = 5  # Nonmax suppress topK to keep


def corner_detect(img, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dest = cv2.cornerHarris(gray, BOCK_SIZE, SOBEL_SIZE, FREE_K)
    dest = nonmax_suppress(dest, NONM_SIZE, NONM_NUM)
    corner_map = dest > CORNER_THRESH * dest.max()
    corner_idx = np.argwhere(corner_map)
    if draw:
        dest = cv2.dilate(dest, None, iterations=3)
        img[dest > CORNER_THRESH * dest.max()] = [0, 255, 0]
    return corner_idx


def nonmax_suppress(corner_map, tile_size, topK):
    assert topK <= tile_size ** 2
    result = np.copy(corner_map)
    h, w = corner_map.shape

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = corner_map[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            sy, sx = tile.shape
            tile_sort = np.argsort(tile, axis=None).reshape(sy, sx)
            thresh = (sx * sy) - topK
            mask = tile_sort >= thresh
            result[y:min(y + tile_size, h), x:min(x + tile_size, w)] *= mask

    return result


def extract_raw_flow(flow_dir, chunk=None):
    flow_files = natsorted(os.listdir(flow_dir))
    if chunk is not None:
        start, end = chunk
        flow_files = flow_files[start:end]
    flow_stack = None
    for i, x in tqdm(enumerate(flow_files)):
        flo = np.load(os.path.join(flow_dir, x))
        if flow_stack is None:
            h, w, c = flo.shape
            b = len(flow_files) if chunk is None else (end - start)
            flow_stack = np.zeros((b, h, w, c))
        flow_stack[i, :, :, :] = flo
    return flow_stack


def get_trace(corner_idx, flow_stack):
    trace = np.zeros((len(corner_idx), len(flow_stack) + 1, 2), dtype=int)
    for i, corner in enumerate(corner_idx):
        trace[i, 0, :] = corner
        cy, cx = float(corner[0]), float(corner[1])
        CY, CX = round(cy), round(cx)

        for f, flo in enumerate(flow_stack):
            dy = flo[CY, CX, 1]
            dx = flo[CY, CX, 0]

            cy = bound(cy + dy, flo.shape[0] - 1)
            cx = bound(cx + dx, flo.shape[1] - 1)
            
            CY, CX = round(cy), round(cx)

            trace[i, f+1, 0] = CY
            trace[i, f+1, 1] = CX
            
    return trace


def bound(val, upper_bound):
    upper_bounded = min(val, upper_bound)
    return max(0, upper_bounded)


def delta_sort(trace, topK):
    trace_deltas = np.zeros(len(trace))
    for i, pt_trace in enumerate(trace):
        diff = pt_trace[1:, :] - pt_trace[:-1, :]
        trace_deltas[i] = np.sqrt(np.square(diff).sum(axis=-1)).sum()

    idxs = np.flip(np.argsort(trace_deltas))[:topK]
    return trace[idxs, :, :]


def rp():
    return np.random.choice(256)


def draw_trace(img, trace):
    for tr in trace:
        b, g, r = rp(), rp(), rp()
        for pt in tr:
            y, x = pt[0], pt[1]
            #img[y, x, :] = [255, 255, 255] #[b, g, r]
            cv2.circle(img, (x, y), 3, (b, g, r), -1)


def np_to_csv(trace, idx=""):
    ypane = trace[:, :, 0]
    xpane = trace[:, :, 1]
    np.savetxt(f"Y_trace{idx}.csv", ypane, fmt="%d", delimiter=",")
    np.savetxt(f"X_trace{idx}.csv", xpane, fmt="%d", delimiter=",")


def draw_partitions(img):
    color = (0, 0, 255)  # b, g, r
    H, W = img.shape[:2]
    tops = [(x, 0) for x in X_SPLITS]
    bottoms = [(x, H - 1) for x in X_SPLITS]
    for t, b in zip(tops, bottoms):
        cv2.line(img, t, b, color, 1)


def get_partitions(img):
    H, W = img.shape[:2]
    lbound = (0, *X_SPLITS)
    rbound = (*X_SPLITS, W)
    return [(l, r) for l, r in zip(lbound, rbound)]


def get_process_chunks(flow_dir, load_steps):
    """ Rounds up for last chunk if not evenly split """
    num_flow = len(os.listdir(flow_dir))
    step_size = int(num_flow / load_steps)
    chunks = []
    for i in range(load_steps):
        chunks.append((i*step_size, (i+1)*step_size))
    
    last_start, last_end = chunks[-1]
    if last_end != num_flow:
        chunks[-1] = (last_start, num_flow)

    return chunks


if __name__ == "__main__":
    img = cv2.imread(image_path)
    print("H, W =", img.shape[0], img.shape[1])

    begin = time.time()

    clean_img = img.copy()

    corner_idx = corner_detect(img, draw=JUST_CORNERS)
    last_corner_idx = corner_idx.copy()

    if not JUST_CORNERS:
        chunks = get_process_chunks(flow_path, LOAD_STEPS)
        traces = []

        for cstart, cend in chunks:
            print(f"Processing chunk ({cstart}, {cend})")
            flow_stack = extract_raw_flow(flow_path, (cstart, cend))
            print("Max Flow =", flow_stack[0].max())
            tr = get_trace(last_corner_idx, flow_stack)
            last_corner_idx = tr[:, -1, :]
            traces.append(tr)
            flow_stack = None
            del flow_stack
        
        trace = np.concatenate(traces, axis=1)
            
    part_imgs = []

    parts = get_partitions(img)
    for pidx, (lbound, rbound) in enumerate(parts):

        if not JUST_CORNERS:
            in_range_idx = np.logical_and(corner_idx[:, -1] >= lbound, corner_idx[:, -1] < rbound)
            sel_traces = trace[in_range_idx, :, :]  # keep only traces with keypoint origin in current partition

            fast_traces = delta_sort(sel_traces, NUM_TRACE)
            draw_trace(img, fast_traces)

            if len(parts) > 0:
                sel_img = clean_img.copy()
                draw_trace(sel_img, fast_traces)
                part_imgs.append(sel_img)
                np_to_csv(fast_traces, pidx)
            else:
                np_to_csv(fast_traces)

        if SHOW_PARTITION:
            draw_partitions(img)

    process_time = time.time() - begin

    print("Time elapsed:", process_time)
    cv2.imwrite("track.jpg", img)
    for pidx, pimg in enumerate(part_imgs):
        cv2.imwrite(f"track{pidx}.jpg", pimg)