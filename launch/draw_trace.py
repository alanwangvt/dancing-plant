"""
Draws circles on points tracked in their
original frames. Then generates an MP4
video to visualize the data.

This assumes trace CSVs have already been generated
using gen_trace.py (which itself requires running
gen_flow.py). These CSVs will be located at
<root>/tracks.

Output CSVs and JPEGs are dumped at <root>/annotated.
All partitions are visualized in one video.

Right now this does not support drawing traces for
the trimmed traces left in <root>/clustered by
cluster_trace.py. But all traces extracted in
those CSVs are included in current results.

Parameters:
    experiment_dir - path to the directory where the input frames are stored (prefix of this is experiment_root in gen_flow.py)
    sample_freq    - sample frequency of frames used, starting with first frame (should match that used in the collections tuple of gen_flow.py)
    circle_size    - radius (in pixels) of circles annotating each tracked point
    fps            - frames-per-second of video generated

Example:
    Say we generated traces of an experiment with frames located at
    '/a/b/c/experiment', so <root>/tracks is populated with trace CSVs.
    The flow for this experiments was generated by skipping every other
    frame (setting '2' in the second field of the collections tuple in 
    gen_flow.py). We want the circles to be six pixels in radius and the
    video to run at ten frame-per-second. We would use...

    experiment_dir = "/a/b/c/experiment"

    sample_freq = 2

    circle_size = 6

    fps = 10

"""

###############################
### MODIFY PARAMETERS BELOW ###

experiment_dir = "/work/alanwang/dataset01/20210621BT/0/"

trace_path = "/home/alanwang/dancing-plant/tracks/"

sample_freq = 5

circle_size = 10

fps = 20

### SHOULD NOT NEED TO MODIFY BELOW ###
#######################################


if __name__ == "__main__":
    from dancing_plant.annotate import run_annotate_with_defaults
    run_annotate_with_defaults(
        experiment_dir,
        trace_path,
        sample_freq,
        circle_size,
        fps=fps
    )
