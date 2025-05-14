import os
import torch
import random
from tqdm import tqdm
from pathlib import Path
from configargparse import ArgumentParser
from models.flow_processor import FlowProcessor


# Extract flow from a dataset without using db


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, help="Path to yaml config file")
    parser.add_argument("--base_path", type=str, help="Path to base directory of data")
    parser.add_argument("--version", type=str, help="version of the dataset")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flow_method", type=str, default="sea_raft", help="Flow estimation method")
    parser.add_argument("--flow_batch_size", type=int, default=4, help="Batch size for flow estimation")
    parser.add_argument("--db_path", type=str, help="Path to database file")
    args, _ = parser.parse_known_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    track_dir = Path(args.base_path) / f"track_{args.version}"
    # track_dir = track_dir/"horse"/"nEzBx0_8mjg_032"
    flow_estimator = FlowProcessor(
        flow_method=args.flow_method,
        image_suffix="rgb.png",
        flow_suffix="flow.png",
        mask_suffix="mask.png",
        flow_batch_size=args.flow_batch_size,
        device=device,
    )
    for cat_dir in track_dir.iterdir():
        print(cat_dir)
        if not cat_dir.is_dir():
            continue
        clip_dirs = list(cat_dir.iterdir())
        random.shuffle(clip_dirs)
        for clip_dir in tqdm(clip_dirs, total=len(os.listdir(cat_dir))):
            if not clip_dir.is_dir():
                continue
            for track_path in clip_dir.iterdir():
                if not track_path.is_dir():
                    continue
                flow_estimator.run(track_path)
