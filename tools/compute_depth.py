import os.path

from configargparse import ArgumentParser
import torch
from pathlib import Path
from random import shuffle
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np


"""Compute depth from a dataset without using db"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to base directory of data")
    parser.add_argument("--depth_method", type=str, default="depth_anything_v2", help="Depth estimation method")
    parser.add_argument("--image_suffix", type=str, default="rgb.png", help="Image file suffix")
    parser.add_argument("--depth_suffix", type=str, default="depth_img.png", help="Depth file suffix")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    if args.depth_method == "depth_anything_v2":
        from models.depth_models.depth_anything_v2 import DepthAnythingV2 as DepthProcessor
        depth_processor = DepthProcessor(device=device)
    else:
        raise NotImplementedError
    all_image_paths = list(Path(args.data_dir).rglob('*' + args.image_suffix))
    shuffle(all_image_paths)
    for image_path in tqdm(all_image_paths):
        if os.path.isfile(str(image_path).replace(args.image_suffix, args.depth_suffix)):
            print(f"{image_path} have been processed")
            continue
        image = Image.open(image_path)
        depth = depth_processor([image])
        depth_path = str(image_path).replace(args.image_suffix, args.depth_suffix)
        depth = depth.detach()
        depth = 65535 * ((depth - torch.amin(depth, dim=(1, 2), keepdim=True)) / (
            torch.amax(depth, dim=(1, 2), keepdim=True) - torch.amin(depth, dim=(1, 2), keepdim=True)
        ))
        depth = depth.cpu().numpy().astype(np.uint16)
        cv2.imwrite(depth_path, depth[0])
