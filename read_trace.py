# Usage: python read_trace.py <path to the dataset> <img sample freq> 
# path to the dataset (e.g., d:\temp\dataset01) is the folder that contains subfolders (e.g., 20210101AT), which contains more subfolders (e.g., 0,1,2,3,4,5)
# the second last subfolder has to be called tracks where X_Trace and Y_Trace files exist
# It can't deal with multiple traces as of yet

import numpy as np
import pandas as pd
import argparse
import glob
import os.path as osp
import re

def processData(xv, yv, dataname):
    outputfile = 'dataOutput.csv'
    dimens = np.shape(xv)
    columnheaders = []
    data = []
    for j in range(0, dimens[0]):        
        row =[]
        minx = 1000
        miny = 1000
        minz = 1000
        maxx=0.0
        maxy=0.0
        maxz=0.0
        if j==0:
            columnheaders.append('class')
        if dataname.find('B')>=0:
            row.append('0')
        elif dataname.find('IA')>=0:
            row.append('1')
        elif dataname.find('A')>=0:
            row.append(2)
        else:
            row.append(dataname)
        for i in range(1, dimens[1]):
            if j==0:
                columnheaders.append(f'x{i}-x{i-1}')
                columnheaders.append(f'y{i}-y{i-1}')
                columnheaders.append(f'z{i}-z{i-1}')
            # print(j, i)
            # print(xv[j,i], xv[j,i-1], yv[j,i], yv[j,i-1])
            xdist = xv[j,i] - xv[j,i-1]
            ydist = yv[j,i] - yv[j,i-1]
            a = np.array((xv[j,i], yv[j,i]))
            b = np.array((xv[j,i-1], yv[j,i-1]))
            zdist = np.linalg.norm(a-b)
            if abs(xdist)<minx:
                minx=abs(xdist)
            if abs(ydist)<miny:
                miny=abs(ydist)
            if abs(zdist)<minz:
                minz=abs(zdist)     
            if abs(xdist)>maxx:
                maxx=abs(xdist)
            if abs(ydist)>maxy:
                maxy=abs(ydist)
            if abs(zdist)>maxz:
                maxz=abs(zdist)        
            row.append(xdist)
            row.append(ydist)
            row.append(zdist)
        # print(columnheaders)
        # print(row)
        if j==0:
            columnheaders.append("minx")
            columnheaders.append("miny")
            columnheaders.append("mine")                
            columnheaders.append("maxx")
            columnheaders.append("maxy")
            columnheaders.append("maxe")
        row.append(minx)
        row.append(miny)
        row.append(minz)
        row.append(maxx)
        row.append(maxy)
        row.append(maxz)            
        data.append(row)   

    df = pd.DataFrame(data=data, columns=columnheaders)
    return df


def saveToCSV(data, csvfilename):
    if not osp.isfile(csvfilename):
        data.to_csv(csvfilename, header=True, sep=',', index=False)
    else: 
        data.to_csv(csvfilename, mode='a', header=False, sep=',', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+', help="the path to the dataset, e.g., /work/alanwang/dataset01")
    parser.add_argument('sample_freq', nargs="+", default=1, help='Image sampling frequency used by gen_flow and annotate')
    args = parser.parse_args()
    # print(args)
    if args.path:
        sample_freq= int(args.sample_freq[0])
        dpath = args.path[0]
        if osp.exists(dpath):
            print(osp.join(dpath, f'**/raft-flow-raw-{sample_freq}/tracks'))
            level1 = glob.glob(osp.join(dpath, f'**/raft-flow-raw-{sample_freq}/tracks'), recursive=True)
            print(level1)
            for i, csvfile in enumerate(level1):
                level2pathlist=re.split('\W+',csvfile)  # or level1path.replace('\\','')  leve1path.replace('.','')
                datasetname = level2pathlist[len(level2pathlist)-8] + "_" + level2pathlist[len(level2pathlist)-7]
                print(osp.join(csvfile, 'X_trace.csv'))
                print(datasetname)
                xdata = np.loadtxt(osp.join(csvfile, 'X_trace.csv'), dtype=np.int64, delimiter=",")
                ydata = np.loadtxt(osp.join(csvfile, 'Y_trace.csv'), dtype=np.int64, delimiter=",")   
                csvfile = f"{level2pathlist[len(level2pathlist)-8]}-{sample_freq}.csv"
                saveToCSV(processData(xdata, ydata, datasetname), csvfile)
                print(csvfile)
