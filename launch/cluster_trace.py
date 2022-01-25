"""
Use time-series clustering to group traces based
on their movement patterns.

This assumes trace CSVs have already been generated
using gen_trace.py (which itself requires running
gen_flow.py). These CSVs will be located at
<root>/tracks.

Output CSVs and JPEGs are dumped at <root>/clustered.
For each partition index <idx>, we generate a image
of each cluster 'track<idx>-K<cluster_idx>.jpg,
and one image of all the extractions, where the top
'num_trace' fastest are taken from each cluster.
If a cluster has less than 'num_trace' traces, all
traces are extracted. 

CSVs follow a similar naming scheme but are labelled
with the number of top traces successfully extracted.

Parameters:
    disp_img_path - image to display generated traces on for visualization purposes (e.g. first image of experiment)
    num_cluster   - number of clusters for each partition; if an integer this value is used for all partitions,
                    otherwise use a tuple with length equal to number of partitions in <root>/tracks
    num_trace     - the number of top traces to keep in each CLUSTER of each partition, sorted by cumulative absolute movement over time

Example:
    Say we generated traces using gen_trace.py for an experiment with two plant
    partitions, and want their traces to be grouped into three and five clusters,
    respectively, after quick visual analysis of the traces. Further, we want
    to keep at most four traces per cluster per partition. And our original
    frames are kept at '/a/b/c/experiment'. We would use...

    disp_img_path = "/a/b/c/experiment/000.jpg"

    num_cluster = (3, 5)

    num_trace = 4

"""

###############################
### MODIFY PARAMETERS BELOW ###

disp_img_path = "/work/alanwang/dataset01/20210623AT/5/5-001.jpg"

num_cluster = 3

num_trace = 5

### SHOULD NOT NEED TO MODIFY BELOW ###
#######################################


if __name__ == "__main__":
    from dancing_plant.cluster import run_cluster_with_defaults
    run_cluster_with_defaults(
        disp_img_path,
        num_cluster,
        num_trace
    )
