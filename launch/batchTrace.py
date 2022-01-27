"""
This script is used to batch process gen_trace, draw_trace, and cluster trace after flows have been generated.
Parameter:
path - The path to the dataset. The path presumably contains two level1 folders, e.g., 20210101AT and 20210101BT. Each level1 folder contains several level2 folders, each of which contains the actual images and a raft-flow folder

Usage:
python launch/batchTrace.py /work/alanwang/dataset01

Result:
Each image folder will contain four subfolders: track, trace_cache, annotated, clustered
"""

import argparse
import os
import dancing_plant
import os.path as osp
import glob
import re
import shutil



if __name__ == '__main__':
    from gen_trace import tBatchTrigger
    from draw_trace import dBatchTrigger
    from cluster_trace import cBatchTrigger
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', help="the path to the dataset, e.g., /work/alanwang/dataset01")
    parser.add_argument('samplefreq', nargs='+', default=1, help='Image sampling frequency used by gen_flow and annotate')
    args = parser.parse_args()
    # print(args)
    if args.path:
        dpath = args.path[0]
        sample_freq = args.samplefreq[0]
        print(sample_freq)
        if os.path.exists(dpath):
            # print(dpath)
            level1 = glob.glob(os.path.join(dpath, '*/'))
            # level1 = os.listdir(dpath)
            # print(level1)
            for i, level1path in enumerate(level1):   # e.g., 20210101BT or 20210102AT
                print(level1path)
                level2 = glob.glob(os.path.join(level1path, '*/'))
                for j, level2path in enumerate(level2):
                    print(level2path)
                    level2pathlist=re.split('\W+',level2path)  # or level1path.replace('\\','')  leve1path.replace('.','')
                    level2pathname = level2pathlist[len(level2pathlist)-2]
                    # print(level1path)
                    # print(level2pathname) # this is supposed to be a number between 0 and 5
                    tBatchTrigger(level1path, level2pathname)
                    dBatchTrigger(level1path, level2pathname)
                    cBatchTrigger(level1path, level2pathname, sample_freq)  

                    # mv trace_cache $1
                    # mv tracks $1
                    # mv clustered $1
                    # mv annotated $1
                    source = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "tracks")
                    dest = osp.join(level1path,level2pathname)
                    shutil.move(source, dest)   
                    source = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "trace_cache")
                    shutil.move(source, dest)            
                    source = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "clustered")
                    shutil.move(source, dest)          
                    source = osp.join(osp.dirname(osp.dirname(dancing_plant.__file__)), "annotated")
                    shutil.move(source, dest)     
                    # print(level1path)

