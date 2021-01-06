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
image_path = "/home/lowell/dancing-plant/DPI/07-27-2020/military-time/Webcam Shot Date July 25 2020 Time 23.52.57.jpg"
flow_path = "/home/lowell/dancing-plant/DPI/07-27-2020/military-time/raft-flow-raw-5"
# image_path = "/home/lowell/dancing-plant/DPI/07-31-2020-Azura/WIN_20200731_19_12_08_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/07-31-2020-Azura/raft-flow-raw-20"
# image_path = "/home/lowell/dancing-plant/DPI/07-15-2020/pictures/WIN_20200715_12_37_37_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/07-15-2020/pictures/raft-flow-raw-10"
# image_path = "/home/lowell/dancing-plant/DPI/selected from 07-22-2020/WIN_20200722_17_23_30_Pro.jpg"
# flow_path = "/home/lowell/dancing-plant/DPI/selected from 07-22-2020/raft-flow-raw-10"

JUST_CORNERS = True  # Will only run corner detection if true, otherwise will run tracking after corner detection

NUM_TRACE = 5  # Top NUM_TRACE most mobile corner traces will be kept

CORNER_THRESH = 0.01  # R scores must be greater than this faction of the max R score
BOCK_SIZE = 2  # Size of local area used when creating R scores from gradients
SOBEL_SIZE = 9  # Size of sobel kernel used in corner detection
FREE_K = 0.00  # Parameter trading off between edge and corner detection (higher is stricter on corner detections, lower will allow more edges)
NONM_SIZE = 40  # Nonmax suppress tile size
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


def extract_raw_flow(flow_dir):
    flow_imgs = [np.load(os.path.join(flow_dir, x)) for x in natsorted(os.listdir(flow_dir))]
    return np.stack(flow_imgs)


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
            cv2.circle(img, (x, y), 6, (b, g, r), -1)


def np_to_csv(trace):
    ypane = trace[:, :, 0]
    xpane = trace[:, :, 1]
    np.savetxt("Y_trace.csv", ypane, fmt="%d", delimiter=",")
    np.savetxt("X_trace.csv", xpane, fmt="%d", delimiter=",")


if __name__ == "__main__":
    img = cv2.imread(image_path)
    print("H, W =", img.shape[0], img.shape[1])

    begin = time.time()

    corner_idx = corner_detect(img, draw=JUST_CORNERS)
    # draw_trace(img, np.expand_dims(corner_idx, 1))

    if not JUST_CORNERS:
        flow_stack = extract_raw_flow(flow_path)
        print(flow_stack[0].max())
        trace = get_trace(corner_idx, flow_stack)
        # print(trace[:, 0:2, :])
        fast_traces = delta_sort(trace, NUM_TRACE)
        draw_trace(img, fast_traces)
        np_to_csv(fast_traces)

    process_time = time.time() - begin

    print("Time elapsed:", process_time)
    cv2.imwrite("track.jpg", img)