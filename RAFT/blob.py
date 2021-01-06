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

from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.color import rgb2hsv

image_path = "/home/lowell/dancing-plant/DPI/07-15-2020/pictures/WIN_20200715_12_37_37_Pro.jpg"
SCALE_REDUCE = 4
QUANTILE = 0.1
HTARGET = 0.6
ERODE_ITR = 5
DILATE_ITR = 5


def mean_shift(img):
    print("orig", img.shape)
    h, w, c = img.shape
    scale = SCALE_REDUCE
    sh = int(h / scale)
    sw = int(w / scale)
    img_small = cv2.resize(img, (sw, sh))
    print("small", img_small.shape)
    hsv = rgb2hsv(img_small)
    print("hsv", hsv.shape)

    # hs = hsv[:, :, :2]
    # print("hs", hs.shape)
    # print(hs.min(), hs.max())

    # hs_flat = hs.flatten().reshape(-1, 2)
    # print("hs_flat", hs_flat.shape)

    hue = hsv[:, :, 0]
    print("hue", hue.shape)
    print(hue.min(), hue.max())

    h_flat = hue.flatten().reshape(-1, 1)
    print("hue_flat", h_flat.shape)

    print("estimating bandwidth")
    bandwidth = estimate_bandwidth(h_flat, quantile=QUANTILE, n_samples=500)
    print("bandwidth:", bandwidth)
    print("clustering...")
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300).fit(h_flat)
    lbl = clustering.labels_
    print(lbl.shape)
    print(lbl.min(), lbl.max())
    print(lbl)

    lbl = lbl.reshape(sh, sw)
    print(lbl.shape)

    # num_clusters = int(lbl.max()) + 1
    # for c in range(num_clusters):
    #     mask = lbl == c
    #     mask = np.stack(3 * [mask], axis=2)
    #     cluster_img = np.copy(img_small)
    #     cluster_img[mask] = 0.
    #     cv2.imwrite("clus{}.jpg".format(c), cluster_img.astype(np.uint8))

    multi_factor = 255.0 / (lbl.max() + 1)
    grey_cluster = (multi_factor * lbl).astype(np.uint8)
    colormap = cv2.applyColorMap(grey_cluster, cv2.COLORMAP_JET)

    # Need to identify which cluster is leaves (circular hue threshold?)
    num_clusters = int(lbl.max()) + 1
    c_sel, c_hue = None, None
    htarget = HTARGET
    for c in range(num_clusters):
        mask = lbl != c
        div = sh * sw - mask.sum()
        # print("num_other", div)
        hue = np.copy(hsv[:, :, 0])
        hue[mask] = 0.
        mean_hue = hue.sum() / div
        # print("mean_hue", mean_hue)
        if c_sel is None or circ_dist(c_hue, htarget) > circ_dist(mean_hue, htarget):
            c_sel = c
            c_hue = mean_hue
        
    print("c_sel", c_sel)
    print("c_hue", c_hue)

    mask = lbl != c_sel
    mask = np.stack(3 * [mask], axis=2)
    leaf_extract = np.copy(img_small)
    leaf_extract[mask] = 0

    # Need to isolate leaves -> try eroding and dialating? Won't work for occluded leaves though
    mask = (lbl == c_sel).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for e in range(1, 11):
        eroded = cv2.erode(mask, kernel, iterations=e)
        dilated = cv2.dilate(eroded, kernel, iterations=e)
        leaf_extract_clean = np.stack(3 * [dilated], axis=2) * img_small
        cv2.imwrite("erode_dilate_{}.jpg".format(e), leaf_extract_clean)



    cv2.imwrite("small.jpg", img_small)
    cv2.imwrite("hue.jpg", (255 * hsv[:, :, 0]).astype(np.uint8))
    cv2.imwrite("sat.jpg", (255 * hsv[:, :, 1]).astype(np.uint8))
    cv2.imwrite("cluster.jpg", colormap)
    cv2.imwrite("extract.jpg", leaf_extract)

    


def circ_dist(a, b):
    if a > b:
        big = a
        small = b
    else:
        small = a
        big = b

    comp1 = big - small
    comp2 = small - (big - 1)
    return min(comp1, comp2)


def write_params(pdict):
    out = json.dumps(pdict)
    with open("params.json", 'w') as f:
        f.write(out)



if __name__ == "__main__":
    img = cv2.imread(image_path)

    begin = time.time()
    mean_shift(img)
    process_time = time.time() - begin

    write_params({
        "scale": SCALE_REDUCE,
        "quantile": QUANTILE,
        "hue_target": HTARGET,
        "erode_itr": ERODE_ITR,
        "dilate_itr": DILATE_ITR,
        "time_elapsed": process_time
    })