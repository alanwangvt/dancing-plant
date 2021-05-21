import sys
sys.path.append('core')

import argparse
import os
import os.path as osp
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import json
import time
import math
import hashlib

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm
from natsort import natsorted, ns

from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2hsv


IMAGE_PATH = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed/000.jpg"
FLOW_PATH = "/mnt/slow_ssd/lowell/DPI/05-04-2021/indexed/raft-flow-raw-1"

FLOW_CHUNK_SIZE = 30  # Max number of flow images that will be loaded and used for tracking at a time

X_SPLITS = ()  # List of X coordinates to partition the track anchors by horizontally
Y_SPLITS = (1000,)  # List of Y coordinates to partition the track anchors by vertically
SHOW_PARTITION = True  # Will draw vertical lines where the tracks are partitioned

JUST_ANCHORS = False  # Will only run anchor generation if true, otherwise will run tracking afterwards

NUM_TRACE = 30  # Top NUM_TRACE most mobile corner traces will be kept

DENSE_TRACK = True  # traces generated in regular grid intervals, otherwise use Harris corner detection params below
GRID_SIZE = 100  # space inbetween dense trace anchors (both x and y), when DENSE_TRACK is True

CORNER_THRESH = 0.00025  # R scores must be greater than this faction of the max R score
BLOCK_SIZE = 2  # Size of local area used when creating R scores from gradients
SOBEL_SIZE = 9  # Size of sobel kernel used in corner detection
FREE_K = 0.00  # Parameter trading off between edge and corner detection (higher is stricter on corner detections, lower will allow more edges)
NONM_SIZE = 80  # Nonmax suppress tile size
NONM_NUM = 5  # Nonmax suppress topK to keep


def corner_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dest = cv2.cornerHarris(gray, BLOCK_SIZE, SOBEL_SIZE, FREE_K)
    dest = nonmax_suppress(dest, NONM_SIZE, NONM_NUM)
    corner_map = dest > CORNER_THRESH * dest.max()
    return np.argwhere(corner_map)


def grid_anchor(img):
    H, W, _ = img.shape
    x_count = np.repeat(np.expand_dims(np.arange(W), axis=1), H, axis=1).T
    y_count = np.repeat(np.expand_dims(np.arange(H), axis=1), W, axis=1)
    coord_map = np.logical_and(x_count % GRID_SIZE == 0, y_count % GRID_SIZE == 0)
    return np.argwhere(coord_map)


def draw_anchors(img, anchors):
    y_idx, x_idx = np.split(anchors, 2, axis=1)
    ravel_idxs = np.ravel_multi_index((y_idx.squeeze(), x_idx.squeeze()), img.shape[:2])

    coord_map = np.zeros(np.prod(img.shape[:2]), dtype=np.bool)  # flattened!
    coord_map[ravel_idxs] = True
    coord_map = coord_map.reshape(img.shape[:2])

    coord_map = cv2.dilate(coord_map.astype(np.float), None, iterations=3)
    img[coord_map > 0.] = [0, 255, 0]  # b, g, r


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
        flo = np.load(osp.join(flow_dir, x))
        if flow_stack is None:
            h, w, c = flo.shape
            b = len(flow_files) if chunk is None else (end - start)
            flow_stack = np.zeros((b, h, w, c))
        flow_stack[i, :, :, :] = flo
    return flow_stack


def get_trace(anchors, flow_stack):
    trace = np.zeros((len(anchors), len(flow_stack) + 1, 2), dtype=int)
    for i, anchor in enumerate(anchors):
        trace[i, 0, :] = anchor
        cy, cx = float(anchor[0]), float(anchor[1])
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

    # horizontal partitions
    tops = [(x, 0) for x in X_SPLITS]
    bottoms = [(x, H - 1) for x in X_SPLITS]
    for t, b in zip(tops, bottoms):
        cv2.line(img, t, b, color, 1)

    # vertical partitions
    lefts = [(0, y) for y in Y_SPLITS]
    rights = [(W - 1, y) for y in Y_SPLITS]
    for l, r in zip(lefts, rights):
        cv2.line(img, l, r, color, 1)


def get_partitions(img):
    H, W = img.shape[:2]

    # horizontal partitions
    lbound = (0, *X_SPLITS)
    rbound = (*X_SPLITS, W)

    # vertical partitions
    tbound = (0, *Y_SPLITS)
    bbound = (*Y_SPLITS, H)

    x_bounds = [(l, r) for l, r in zip(lbound, rbound)]
    y_bounds = [(t, b) for t, b in zip(tbound, bbound)]

    grid_bounds = []
    for yb in y_bounds:
        t, b = yb
        grid_row = [(l, r, t, b) for l, r in x_bounds]
        grid_bounds += grid_row

    return grid_bounds


def get_process_chunks(flow_dir, flow_chunk_size):
    """Rounds up for last chunk if not evenly split."""
    num_flow = len(os.listdir(flow_dir))
    load_steps = math.ceil(num_flow / flow_chunk_size)
    chunks = []
    for i in range(load_steps):
        chunks.append((i*flow_chunk_size, (i+1)*flow_chunk_size))
    
    last_start, last_end = chunks[-1]
    if last_end != num_flow:
        chunks[-1] = (last_start, num_flow)

    return chunks


def get_working_hash():
    """
    Return first 8 hex chars of a hash of the image and flow paths
    plus any relevant anchor generation parameters.
    Alows storing a sampled trace numpy array for further
    post-processing (e.g. find_best_traces) without constantly
    reloading the dense flow files, which are large.
    """
    if DENSE_TRACK:
        PARAM_STR = str(GRID_SIZE)
    else:
        PARAM_STR = str(CORNER_THRESH) + \
            str(BLOCK_SIZE) + \
            str(SOBEL_SIZE) + \
            str(FREE_K) + \
            str(NONM_SIZE) + \
            str(NONM_NUM)

    string = bytearray(IMAGE_PATH + FLOW_PATH + PARAM_STR, "utf8")
    return hashlib.sha1(string).hexdigest()[:8]


def working_dir():
    wdir = osp.join(osp.dirname(__file__), "trace_cache")
    if not osp.exists(wdir):
        os.makedirs(wdir)
    return wdir


def have_working_trace(thash):
    file_names = os.listdir(working_dir())
    for fn in file_names:
        if fn.startswith("trace_" + thash):
            return True
    return False


def save_working_trace(thash, trace):
    out_path = osp.join(working_dir(), "trace_" + thash + ".npy")
    np.save(out_path, trace)


def load_working_trace(thash):
    in_path = osp.join(working_dir(), "trace_" + thash + ".npy")
    return np.load(in_path)


def trace_from_flow(flow_path, anchors):
    chunks = get_process_chunks(flow_path, FLOW_CHUNK_SIZE)
    last_track = anchors.copy()
    traces = []

    for cstart, cend in chunks:
        print(f"Processing chunk ({cstart}, {cend})")
        flow_stack = extract_raw_flow(flow_path, (cstart, cend))
        # print("Max Flow =", flow_stack[0].max())
        tr = get_trace(last_track, flow_stack)
        last_track = tr[:, -1, :]
        tr_tail = tr if cstart == 0 else tr[:, 1:, :]  # removes redundant starting point after first chunk
        traces.append(tr_tail)
        flow_stack = None
        del flow_stack
    
    return np.concatenate(traces, axis=1)


def find_best_traces(trace, parts):
    """Based on most cumulative displacement."""
    fast_traces_list = []
    anch = trace[:, 0, :]  # extract anchors from trace
    for pidx, part in enumerate(parts):
        lbound, rbound, tbound, bbound = part

        in_range_X = np.logical_and(anch[:, -1] >= lbound, anch[:, -1] < rbound)
        in_range_Y = np.logical_and(anch[:, 0] >= tbound, anch[:, 0] < bbound)
        in_range_idx = np.logical_and(in_range_X, in_range_Y)

        sel_traces = trace[in_range_idx, :, :]  # keep only traces with keypoint origin in current partition

        fast_traces = delta_sort(sel_traces, NUM_TRACE)
        fast_traces_list.append(fast_traces)
    
    return fast_traces_list


def save_traces(traces_list, img):
    """Writes each trace array to a csv. Also draws traces."""
    part_imgs = []
    clean_img = img.copy()
    num_parts = len(traces_list)
    
    for pidx, trace in enumerate(traces_list):
        draw_trace(img, trace)  # img aggregates all trace drawings
        
        if num_parts > 1:  # also save partition-specific traces
            sel_img = clean_img.copy()
            draw_trace(sel_img, trace)
            part_imgs.append(sel_img)
            np_to_csv(trace, pidx)
        else:
            np_to_csv(trace)

    return part_imgs  # will be empty list if num_parts == 1 (i.e. whole image) since img is already annotated



if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    print("H, W =", img.shape[0], img.shape[1])

    begin = time.time()

    thash = get_working_hash()
    using_saved_trace = have_working_trace(thash)

    if using_saved_trace:  # can load exisiting anchors and trace
        print("Loading stored intermediate trace from trace_cache...")
        trace = load_working_trace(thash)
        anchors = trace[:, 0, :]
    else:  # must generate using RAFT flow files
        anchors = corner_detect(img) if not DENSE_TRACK else grid_anchor(img)

    if not JUST_ANCHORS:
        if not using_saved_trace:
            trace = trace_from_flow(FLOW_PATH, anchors)
            save_working_trace(thash, trace)

        parts = get_partitions(img)
        fast_traces_list = find_best_traces(trace, parts)
        
        part_imgs = save_traces(fast_traces_list, img)
        for pidx, pimg in enumerate(part_imgs):
            cv2.imwrite(f"track{pidx}.jpg", pimg)
    else:
        draw_anchors(img, anchors)

    if SHOW_PARTITION:
        draw_partitions(img)

    cv2.imwrite("track.jpg", img)
    
    process_time = time.time() - begin
    print("Time elapsed:", process_time)