
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from attrdict import AttrDict

from dancing_plant.raft.raft import RAFT
from dancing_plant.raft.utils import flow_viz
from dancing_plant.raft.utils.utils import InputPadder


EXPERIMENTS = [
    ("05-18-2021a/indexed", 1)
]


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def save_flow(path, flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)

    cv2.imwrite(path, flo)


def save_flow_raw(path, flo):
    flo = flo[0].permute(1,2,0).cpu().numpy()

    prefix = ''.join(path.split('.')[:-1])
    print(prefix)
    np.save(prefix, flo)


def flow(args, collections):
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
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

            print(f"Running RAFT on {input_path} with sample frequency {frameskip}")
            
            images = glob.glob(os.path.join(input_path, '*.png')) + \
                     glob.glob(os.path.join(input_path, '*.jpg'))

            images = sorted(images)
            kept_images = [images[i] for i in range(0, len(images), frameskip)]
            images = kept_images

            for idx, (imfile1, imfile2) in tqdm(enumerate(zip(images[:-1], images[1:]))):
                image1 = load_image(imfile1).to(device)
                image2 = load_image(imfile2).to(device)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                
                ext = imfile1.split('.')[-1]
                out_name = f"{idx}.{ext}"
                specific_path = os.path.join(output_path, out_name)

                save_flow_raw(specific_path, flow_up)


def run_flow_with_defaults(collect_path, collections, model_path, save_prefix="", device=0):
    args = AttrDict({
        "path": collect_path,
        "save_prefix": save_prefix,
        "device": device,
        "model": model_path,
        "alternate_corr": True,
        "small": False,
        "mixed_precision": False
    })

    flow(args, collections)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help="gpu index to use", default=0)
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--save_prefix', help='path prefix of where to save flow arrays, will default to input path if not given', default=None)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    flow(args, EXPERIMENTS)
