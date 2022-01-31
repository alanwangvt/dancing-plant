"""
Generate dense optical flow maps using RAFT.

First, place all experiment images in separate 
directories at common prefix, for example:
    'a/b/c/exp1'
    'a/b/c/exp2'
    etc.

Each directory should contain either JPEGs or
PNGs which already sort by name into the 
time-correct order. See imsort.py for help
with this.

Outputs dense flow maps as .npy files. Will
produce one map for each adjacent set of frames,
ultimately producing NUM_FRAMES - 1 maps.

Parameters:
    experiment_root - root path to all experiments (e.g. 'a/b/c' above)
    collections     - list of experiment tuples in format (directory, sample-frequency)
    save_prefix     - root path output saving directory (defaults to experiment root if empty or None)
    device          - CUDA index for GPU to use

Example:
    Say you wanted to generate flow maps for two image directories
    'exp1' and 'exp2' located at '/a/b/c' using sampling every 1 and 5 frames
    over time, respectively. And you want to save the flow maps to '/mnt/hdd' 
    and use GPU index 1. You would use the following...

    experiment_root = "/a/b/c"

    collections = [
        ("exp1", 1),
        ("exp2", 5)
    ]

    save_prefix = "/mnt/hdd"

    device = 1

"""

###############################
### MODIFY PARAMETERS BELOW ###

experiment_root = "/work/alanwang/"

collections = [
    ("dataset01/20210621BT/0", 7),
    ("dataset01/20210621BT/1", 7),
    ("dataset01/20210621BT/2", 7),
    ("dataset01/20210621BT/3", 7),
    ("dataset01/20210621BT/4", 7),
    ("dataset01/20210621BT/5", 7),
    ("dataset01/20210623AT/0", 7),
    ("dataset01/20210623AT/1", 7),
    ("dataset01/20210623AT/2", 7),
    ("dataset01/20210623AT/3", 7),
    ("dataset01/20210623AT/4", 7),
    ("dataset01/20210623AT/5", 7),
    ("dataset01/20210621BT/0", 8),
    ("dataset01/20210621BT/1", 8),
    ("dataset01/20210621BT/2", 8),
    ("dataset01/20210621BT/3", 8),
    ("dataset01/20210621BT/4", 8),
    ("dataset01/20210621BT/5", 8),
    ("dataset01/20210623AT/0", 8),
    ("dataset01/20210623AT/1", 8),
    ("dataset01/20210623AT/2", 8),
    ("dataset01/20210623AT/3", 8),
    ("dataset01/20210623AT/4", 8),
    ("dataset01/20210623AT/5", 8),
    ("dataset01/20210621BT/0", 9),
    ("dataset01/20210621BT/1", 9),
    ("dataset01/20210621BT/2", 9),
    ("dataset01/20210621BT/3", 9),
    ("dataset01/20210621BT/4", 9),
    ("dataset01/20210621BT/5", 9),
    ("dataset01/20210623AT/0", 9),
    ("dataset01/20210623AT/1", 9),
    ("dataset01/20210623AT/2", 9),
    ("dataset01/20210623AT/3", 9),
    ("dataset01/20210623AT/4", 9),
    ("dataset01/20210623AT/5", 9)       
]

save_prefix = "/work/alanwang/"

device = 0

### SHOULD NOT NEED TO MODIFY BELOW ###
#######################################


def get_model_path():
    model_path = "models/raft-things.pth"  # relative path
    import os.path as osp
    if not osp.exists(model_path):
        raise EnvironmentError("Need to download RAFT model."
            "Run 'download_raft_model.sh' in project root.")
    return model_path


if __name__ == "__main__":
    from dancing_plant.flow import run_flow_with_defaults
    run_flow_with_defaults(
        experiment_root,
        collections,
        get_model_path(),
        save_prefix,         ### save_prefix=save_prefix,
        device        ###   device=device
    )