import os
from configargparse import ArgumentParser
import shutil
from pathlib import Path
from tqdm import tqdm
from random import shuffle
from database import parse_version
from models.dataset_builder import DatasetBuilder

from tools.data_ids import filter_video_ids, filter_clip_ids, filter_track_ids, keep_track_ids


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, help="Path to yaml config file")
    parser.add_argument("--base_path", type=str, help="Path to base directory of data")
    parser.add_argument("--db_path", type=str, help="Path to database file")
    parser.add_argument("--version", type=str, help="version of the dataset")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting")
    parser.add_argument("--train_proportion", type=float, default=0, help="Proportion of training set")
    parser.add_argument(
        "--file_suffixes", type=str, nargs='+', default=[
            "rgb.png", "occlusion.png", "mask.png", "flow.png", "depth.png", "metadata.json", "track_info.json",
            "keypoint.txt", "pose.png"
        ],
        help="File suffix of the data to copy"
    )
    parser.add_argument(
        "--occlusion_filter_threshold", type=float, default=1.,  # 0.1
        help="Threshold for filtering tracks or frames with high occlusion, 1 for no filtering"
    )
    parser.add_argument(
        "--filter_method", default="track", choices=["None", "track", "frame"],
        help=(
            "Method to filter tracks or frames with high occlusion or low flow, "
            "None for no filtering and copy all tracks, "
            "track for filtering whole track based on average occlusion/flow (fast), "
            "frame for filtering based on frame occlusion/flow, finding longest consecutive frames,"
            "considering min_scene_len and max_track_gap (slow)"
        )
    )
    parser.add_argument(
        "--flow_filter_threshold", type=float, default=0,  # 1
        help="Threshold for filtering tracks with low flow, 0 for no filtering"
    )
    parser.add_argument("--min_scene_len", type=float, default=20, help="Minimum length of a track")
    parser.add_argument(
        "--max_track_gap", type=int, default=5,
        help=(
            "Maximum gap of a track between two consecutive frames to consider as the same track, "
            "used in removing discontinuous tracks and final smoothing"
        )
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_version = parse_version(args.version).get("track")
    dataset_name = f"data_{data_version}" if data_version else "data"
    dataset_dir = Path(args.base_path) / dataset_name
    dataset_builder = DatasetBuilder(
        dataset_dir=dataset_dir,
        db_path=args.db_path,
        version=args.version,
        train_proportion=args.train_proportion,
        file_suffixes=args.file_suffixes,
        filter_method=args.filter_method,
        occlusion_filter_threshold=args.occlusion_filter_threshold,
        flow_filter_threshold=args.flow_filter_threshold,
        min_scene_len=args.min_scene_len,
        seed=args.seed,
        verbose=args.verbose,
        # filter_video_ids=filter_video_ids,
        # filter_clip_ids=filter_clip_ids,
        # filter_track_ids=filter_track_ids,
        keep_track_ids=keep_track_ids,
    )
    dataset_builder.run()




