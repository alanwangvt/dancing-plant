"""
Use time-series clustering to find decorrelated traces
"""

import os
import os.path as osp
import numpy as np
import cv2
from tslearn.clustering import TimeSeriesKMeans

from track import draw_trace

trace_path = "/home/lowell/dancing-plant/RAFT"  # assumes traces in format X_trace{i}.csv and Y_trace{i}.csv where 'i' is partition
out_path = "/home/lowell/dancing-plant/RAFT/dtw_corr_traces"
image_path = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed/000.jpg"  # for extraction display

NUM_PART = 2
NUM_CLUSTER = (4, 4)  # num clusters for each k-means run (one for each partition)
NUM_TRACE = 3  # Top NUM_TRACE most mobile corner traces will be kept for EACH cluster


if __name__ == "__main__":
    
    # load trace
    if NUM_PART is None or NUM_PART <= 1:
        x_split_paths = [osp.join(trace_path, "X_trace.csv")]
        y_split_paths = [osp.join(trace_path, "Y_trace.csv")]
    else:
        x_split_paths = [osp.join(trace_path, f"X_trace{idx}.csv") for idx in range(NUM_PART)]
        y_split_paths = [osp.join(trace_path, f"Y_trace{idx}.csv") for idx in range(NUM_PART)]

    x_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in x_split_paths]
    y_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in y_split_paths]

    # load image for visualization purposes
    img = cv2.imread(image_path)

    # make out path if needed
    if not osp.exists(out_path):
        os.makedirs(out_path)

    # time-series cluster each x split with dynamic time warping
    # this scales horribly with both number of time series and time series length, so shouldn't use more than maybe 25 traces per split
    for idx, pack in enumerate(zip(NUM_CLUSTER, x_split_traces, y_split_traces)):
        K, xt, yt = pack

        # create xy-coordinate stack
        xyt = np.stack((yt, xt), axis=-1)
        n_ts, sz, d = xyt.shape

        # translate all traces to start at (0, 0) since we care about delta correlation
        xyt = xyt.astype(np.int32)
        start_pos = np.expand_dims(xyt[:, 0, :], axis=1)
        offset = np.repeat(start_pos, sz, axis=1)
        xyt -= offset

        # fit time-series k-means model to traces (NOTE: expensive)
        model = TimeSeriesKMeans(n_clusters=K, metric="dtw", max_iter=10)
        model.fit(xyt)

        # extract fastest moving trace for each cluster
        extractions = np.empty_like(xyt, shape=(K, NUM_TRACE, sz, d))
        cluster_labels = model.predict(xyt)
        xyt += offset  # fix offset
        for k in range(K):
            cluster_traces = xyt[cluster_labels == k]
            extractions[k, :, :, :] = cluster_traces[:NUM_TRACE]  # take top NUM_TRACE fastest moving traces

            # save extracted traces
            ex_xt = extractions[k, :, :, 1]
            ex_yt = extractions[k, :, :, 0]
            np.savetxt(osp.join(out_path, f"Y_trace{idx}-K{k}-top{NUM_TRACE}.csv"), ex_yt, fmt="%d", delimiter=",")
            np.savetxt(osp.join(out_path, f"X_trace{idx}-K{k}-top{NUM_TRACE}.csv"), ex_xt, fmt="%d", delimiter=",")

            # display full cluster
            img_k = img.copy()
            draw_trace(img_k, cluster_traces)
            cv2.imwrite(osp.join(out_path, f"track{idx}-K{k}.jpg"), img_k)

        # display all extracted traces together
        img_e = img.copy()
        extractions = extractions.reshape(-1, sz, d)  # flatten cluster and trace idx dims
        draw_trace(img_e, extractions)
        cv2.imwrite(osp.join(out_path, f"track{idx}-all-top{NUM_TRACE}-each.jpg"), img_e)
