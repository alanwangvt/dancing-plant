import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

from tqdm import tqdm

# collections = [("07-15-2020/pictures", 10),
#         ("07-27-2020/military-time", 5),
#         ("07-31-2020-Azura", 20),
#         ("selected07232020AZURA", 10),
#         ("selected from 07-22-2020", 10),
#         ("selected from 07-23-2020_USB", 10),
#         ("selected pictures from 07-21-2020", 5)
# ]

# collections = [("11-01-2020_Azura/military-time", 5),
#         ("11-03-2020Azura/military-time", 10),
#         ("11-01-2020_Azura/military-time", 3),
#         ("11-01-2020_Azura/military-time", 10),
#         ("11-03-2020Azura/military-time", 5),
#         ("11-03-2020Azura/military-time", 20),
# ]

# collections = [("01-24-2021/military-time", 1)
# ]

# collections = [("02-01-2021/military-time", 1)
# ]

collections = [
    ("05-04-2021/indexed", 1),
    ("05-08-2021/indexed", 1),
    ("05-18-2021a/indexed", 1)
]
        


DEVICE = 'cuda:1'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def save_flow(path, flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)

    cv2.imwrite(path, flo)


def save_flow_raw(path, flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()

    ### prefix = ".." + ''.join(path.split('.')[:-1])
    prefix = ''.join(path.split('.')[:-1])
    np.save(prefix, flo)

    # split = path.split('.')
    # prefix = ''.join(split[:-1])
    # ext = split[-1]
    # path_u = prefix + "u." + ext
    # path_v = prefix + "v." + ext
    # print(path_u)
    # print(path_v)
    # print(flo[:,:,0].shape)
    # print(flo[:,:,0].dtype)

    # cv2.imwrite(path_u, flo[:,:,0])
    # cv2.imwrite(path_v, flo[:,:,1])


def demo(args):
    torch.cuda.set_device(1)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():

        for folder, frameskip in collections:
            input_path = os.path.join(args.path, folder)

            if args.save_prefix is not None:
                output_path = os.path.join(args.save_prefix, folder, f"raft-flow-raw-{frameskip}")
            else:  # default to saving in directory off input path
                output_path = os.path.join(input_path, f"raft-flow-raw-{frameskip}")
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            elif os.listdir(output_path):
                print(f"Skipping {input_path}, {output_path} exists and is populated...")
                continue

            print(f"Running RAFT on {input_path}")
            
            images = glob.glob(os.path.join(input_path, '*.png')) + \
                     glob.glob(os.path.join(input_path, '*.jpg'))

            images = sorted(images)
            kept_images = [images[i] for i in range(0, len(images), frameskip)]
            images = kept_images

            for idx, (imfile1, imfile2) in tqdm(enumerate(zip(images[:-1], images[1:]))):
                image1 = load_image(imfile1)
                image2 = load_image(imfile2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                #print(flow_low.shape, flow_up.shape, image1.shape)

                #im1prfx = os.path.basename(imfile1).split('.')[0]
                #im2prfx = os.path.basename(imfile2).split('.')[0]
                
                ext = imfile1.split('.')[-1]
                out_name = f"{idx}.{ext}"
                specific_path = os.path.join(output_path, out_name)

                save_flow_raw(specific_path, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--save_prefix', help='path prefix of where to save flow arrays, will default to input path if not given', default=None)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
