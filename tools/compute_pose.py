# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import torch

from models.pose_processor import PoseProcessor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--pose_method", type=str, default="vitpose", help="Pose estimation method")
    parser.add_argument("--pose_batch_size", type=int, default=1, help="Batch size for flow estimation")
    parser.add_argument("--pose_image_suffix", type=str, default="pose.png", help="Pose image suffix")
    parser.add_argument("--keypoint_suffix", type=str, default="keypoint.txt", help="Pose file suffix")
    parser.add_argument("--image_suffix", type=str, default="rgb.png", help="Image file suffix")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_processor = PoseProcessor(
        pose_method=args.pose_method,
        image_suffix=args.image_suffix,
        keypoint_suffix=args.keypoint_suffix,
        pose_image_suffix=args.pose_image_suffix,
        batch_size=args.pose_batch_size,
        device=device,
        save_pose_image=True,
    )
    pose_processor.run_track(args.data_dir)
