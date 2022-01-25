"""
Generates trace CSVs and corresponding JPEG
image displays using flow maps (such as those
generated from gen_flow.py).

Top-level process:

    1) Anchor points are generated over the entire
       image in a mesh at a specified grid width.

    2) Tracking is split into distinct partitions
       based on specified x and y splits. (Anchors
       which lie within a partition will be tracked
       as part of that partition throughout the video,
       even if they eventually leave that region.)

    3) Anchors are tracked over time using the flow
       maps in order to create CSVs.

    4) The fastest moving traces, sorted by cumulative
       movement in-between each flow file, are extracted
       for each partition. This removes noise and isolates
       the "most interesting" traces.

Output CSVs and JPEGs will appear in <root>/tracks
where <root> is the root directory of this project.
The 'tracks' directory is created automatically.

Several independent trace batches can be generated by dissecting
the image using x_splits and y_splits, visualized in red by setting
show_partition = True. In this case, a CSV and JPEG is created for
the origin anchors within each grid cell, indexed left-to-right
top-to-bottom (reading order) starting at 0.

If just_anchors = True, only the origin points for the
track will be displayed on track.jpg, but flow maps will
not be loaded and traces will not be generated. This is
useful as a sort of "dry-run" tuning the grid_size,
x_splits, and y_splits parameters.

To avoid redundant reloading of flow map files,
which can often be very large, intermediate
trace representations are stored in <root>/trace_cache
wherever possible. It's fine to delete this directory
as desired.

Assuming just_anchors = False, modifying the following 
parameters WILL require reloading flow maps:
    flow_dir
    disp_img_path
    grid_size

Modifying the following parameters will NOT require
reloading flow maps and will instead use a cached
trace, if available:
    show_partition
    x_splits
    y_splits
    num_trace

Parameters:
    just_anchors    - True to only generate and display origin points for trace (and maybe partitions),
                      False to generate origins and subsequent traces from flow
    show_partition  - True to show grid partitions in red, False to omit from display
    flow_dir        - path to the directory of flow map files (0.npy, 1.npy, ...)
    disp_img_path   - image to display generated traces on for visualization purposes (e.g. first image of experiment)
    x_splits        - tuple of X (width) coordinates to dissect experiments by, leave empty () for no partition and include comma (123,) if only one
    y_splits        - tuple of Y (height) coordinates to dissect experiments by, leave empty () for no partition and include comma (123,) if only one
    num_trace       - the number of top traces to keep in each partition, sorted by cumulative absolute movement over time
    grid_size       - length in-between anchors, the origin points of tracking (setting this to '1' would track every pixel; this is expensive) 

Example:
    Say we wanted to generate trace files for an experiment with two plants,
    one on the left and one of the right, and the image dimensions were
    (H, W) = (2000, 3000). And the flow maps were located on a mounted
    hardrive at '/mnt/hdd/experiment/raft-flow-raw-1' the corresponding
    images at '/a/b/c/experiment'. One reasonable configuration would be...

    just_anchors = False
    show_partition = True

    flow_dir = "/mnt/hdd/experiment/raft-flow-raw-1"

    disp_img_path = "/a/b/c/experiment/000.jpg"

    x_splits = (1500,)
    y_splits = ()

    num_trace = 30

    grid_size = 100

"""

###############################
### MODIFY PARAMETERS BELOW ###

just_anchors = False
show_partition = True

flow_dir = "/work/alanwang/dataset01/20210623AT/3/raft-flow-raw-1"

disp_img_path = "/work/alanwang/dataset01/20210623AT/3/3-001.jpg"

x_splits = ()
y_splits = ()

num_trace = 30

grid_size = 100

### SHOULD NOT NEED TO MODIFY BELOW ###
#######################################


if __name__ == "__main__":
    from dancing_plant.track import run_track_with_defaults
    run_track_with_defaults(
        disp_img_path,
        flow_dir,
        x_splits,
        y_splits,
        show_partition,
        just_anchors,
        num_trace,
        grid_size
    )
