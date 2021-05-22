"""
Use time-series clustering to find decorrelated traces
"""

import os
import os.path as osp
import numpy as np
import cv2
from tslearn.clustering import TimeSeriesKMeans

import dancing_plant
from dancing_plant.track import draw_trace
from dancing_plant.track import out_pfx as track_out_pfx

TRACE_PATH = "/home/lowell/dancing-plant/RAFT"  # assumes traces in format X_trace{i}.csv and Y_trace{i}.csv where 'i' is partition
OUT_PATH = "/home/lowell/dancing-plant/RAFT/dtw_corr_traces"
IMAGE_PATH = "/home/lowell/dancing-plant/DPI/05-04-2021/indexed/000.jpg"  # for extraction display

NUM_PART = 2
NUM_CLUSTER = (4, 4)  # num clusters for each k-means run (one for each partition)
NUM_TRACE = 3  # Top NUM_TRACE most mobile corner traces will be kept for EACH cluster


def cluster(trace_path, out_path, image_path, num_part, num_cluster, num_trace):
    # load trace
    if num_part is None or num_part <= 1:
        x_split_paths = [osp.join(trace_path, "X_trace.csv")]
        y_split_paths = [osp.join(trace_path, "Y_trace.csv")]
    else:
        x_split_paths = [osp.join(trace_path, f"X_trace{idx}.csv") for idx in range(num_part)]
        y_split_paths = [osp.join(trace_path, f"Y_trace{idx}.csv") for idx in range(num_part)]

    x_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in x_split_paths]
    y_split_traces = [np.loadtxt(fname, dtype=np.uint32, delimiter=",") for fname in y_split_paths]

    # load image for visualization purposes
    img = cv2.imread(image_path)

    # make out path if needed
    if not osp.exists(out_path):
        os.makedirs(out_path)

    # time-series cluster each x split with dynamic time warping
    # this scales horribly with both number of time series and time series length, so shouldn't use more than maybe 25 traces per split
    for idx, pack in enumerate(zip(num_cluster, x_split_traces, y_split_traces)):
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
        extract_list = []
        cluster_labels = model.predict(xyt)
        xyt += offset  # fix offset
        for k in range(K):
            cluster_traces = xyt[cluster_labels == k]
            num_extract = min(num_trace, len(cluster_traces))

            extractions = cluster_traces[:num_extract]  # take top fastest moving traces
            extract_list.append(extractions)

            # save extracted traces
            ex_xt = extractions[:, :, 1]
            ex_yt = extractions[:, :, 0]
            np.savetxt(osp.join(out_path, f"Y_trace{idx}-K{k}-top{num_extract}.csv"), ex_yt, fmt="%d", delimiter=",")
            np.savetxt(osp.join(out_path, f"X_trace{idx}-K{k}-top{num_extract}.csv"), ex_xt, fmt="%d", delimiter=",")

            # display full cluster
            img_k = img.copy()
            draw_trace(img_k, cluster_traces)
            cv2.imwrite(osp.join(out_path, f"track{idx}-K{k}.jpg"), img_k)

        # display all extracted traces together
        img_e = img.copy()
        all_extractions = np.concatenate(extract_list, axis=0)
        draw_trace(img_e, all_extractions)
        cv2.imwrite(osp.join(out_path, f"track{idx}-all-tops.jpg"), img_e)


def out_pfx(file_name):
    """Prepend file name with fixed output prefix."""
    pfx = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "clustered")
    if not osp.exists(pfx):
        os.makedirs(pfx)
    return osp.join(pfx, file_name)


def get_num_part(trace_path):
    file_names = os.listdir(trace_path)
    trace_key = "_trace"
    trace_names = filter(lambda s: s[1:].startswith(trace_key), file_names)
    part_idxs = [int(x.split(trace_key)[-1][0]) for x in trace_names]
    return max(part_idxs) + 1


def run_cluster_with_defaults(image_path, num_cluster, num_trace):
    trace_path = track_out_pfx("")

    if not os.listdir(trace_path):
        raise EnvironmentError("No trace CSVs found at expected trace path: "
            f"{trace_path}\nRun gen_flow.py and gen_trace.py first.")

    num_part = get_num_part(trace_path)

    if type(num_cluster) is int:
        num_cluster = [num_cluster] * num_part

    if num_part != len(num_cluster):
        raise ValueError(f"Found {num_part} partitions, but only "
            f"specified {len(num_cluster)} cluster quantities. Need "
            f"one cluster quantity per partition.")

    cluster(
        trace_path, 
        out_pfx(""), 
        image_path, 
        num_part, 
        num_cluster, 
        num_trace
    )



if __name__ == "__main__":
    cluster(
        TRACE_PATH, 
        OUT_PATH, 
        IMAGE_PATH, 
        NUM_PART, 
        NUM_CLUSTER, 
        NUM_TRACE
    )
    
