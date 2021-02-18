
import os
import os.path as osp
import cv2
import glob
import numpy as np

from tqdm import tqdm
from natsort import natsorted, ns

from corner_track import rp

image_path = "/home/lowell/dancing-plant/DPI/02-01-2021/military-time/indexed"
trace_path = "/home/lowell/dancing-plant/RAFT"  # assumes traces in format X_trace{i}.csv and Y_trace{i}.csv where 'i' is partition

NUM_X_SPLITS = 2


if __name__ == "__main__":

    # get all images in stack
    image_paths = glob.glob(osp.join(image_path, "*.jpg")) + \
                  glob.glob(osp.join(image_path, "*.png"))
    image_paths = natsorted(image_paths)

    image_paths = image_paths[:100]  ### TEMPORARY WHILE NO MEM

    img0 = cv2.imread(image_paths[0])
    H, W, _ = img0.shape
    images = np.zeros((len(image_paths), H, W, 3), dtype=np.uint8)

    print(f"Loading {len(images)} images from {image_path}")
    for i, ipath in tqdm(enumerate(image_paths[1:])):
        images[i, :, :, :] = cv2.imread(ipath)
    
    # load trace
    if NUM_X_SPLITS is None:
        x_split_paths = [osp.join(trace_path, "X_trace.csv")]
        y_split_paths = [osp.join(trace_path, "Y_trace.csv")]
    else:
        x_split_paths = [osp.join(trace_path, f"X_trace{idx}.csv") for idx in range(NUM_X_SPLITS)]
        y_split_paths = [osp.join(trace_path, f"Y_trace{idx}.csv") for idx in range(NUM_X_SPLITS)]

    x_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in x_split_paths]
    y_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in y_split_paths]

    # draw trace points on each image (same color for each trace)
    tcolor = [dict() for _ in range(NUM_X_SPLITS)]
    for split_idx in range(NUM_X_SPLITS):
        for i in range(len(x_split_traces[split_idx])):  # y_split_traces should be same length
            b, g, r = rp(), rp(), rp()
            tcolor[split_idx][i] = (b, g, r)

    for split_idx in range(NUM_X_SPLITS):
        xtr = x_split_traces[split_idx]
        ytr = y_split_traces[split_idx]
        tcol = tcolor[split_idx]

        for trace_idx in range(len(xtr)):
            b, g, r = tcol[trace_idx]

            for img_idx in range(xtr.shape[-1]):
                if img_idx >= 100:  ### TEMPORARY WHILE NO MEM
                    break
                x = xtr[trace_idx, img_idx]
                y = ytr[trace_idx, img_idx]
                cv2.circle(images[img_idx, :, :, :], (x, y), 3, (b, g, r), -1)

    # save image stack to directory (use ffmpeg from there)
    save_path = osp.join(image_path, "annotated")
    if not osp.exists(save_path):
        os.mkdir(save_path)

    print(f"Writing annotated images to {save_path}")
    for i in tqdm(range(len(images))):
        fout_path = osp.join(save_path, osp.basename(image_paths[i]))
        cv2.imwrite(fout_path, images[i])