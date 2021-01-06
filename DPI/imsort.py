import sys

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from shutil import copyfile



def get_file_sort(args):
    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))
             
    images = sorted(images)
    for imp in images:
        print(imp)


def revise(args):
    images = glob.glob(os.path.join(args.path, '*.png')) + \
             glob.glob(os.path.join(args.path, '*.jpg'))
             
    out_path = os.path.join(args.path, "military-time")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for imp in images:
        tokens = imp.split(' ')
        phase, ext = tokens[-1].split('.')
        hour, mint, sec = tokens[-2].split('.')

        hour = int(hour)
        if phase == "PM" and hour != 12:
            hour += 12
        elif phase == "AM" and hour == 12:
            hour = 0
        if hour < 10:
            hour = '0' + str(hour)
        else:
            hour = str(hour)

        mtime = ".".join([hour, mint, sec])
        ext = "." + ext
        ntokens = tokens[:-2] + [mtime]
        nimp = " ".join(ntokens) + ext

        file_path = os.path.join(out_path, os.path.basename(nimp))
        print(file_path)
        copyfile(imp, file_path)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="dataset for evaluation")
    parser.add_argument("--revise", "-r", action="store_true", help="revise to sortable military time")
    args = parser.parse_args()

    if args.revise:
        revise(args)
    else:
        get_file_sort(args)
