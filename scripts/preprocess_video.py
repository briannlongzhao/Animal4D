import os
import sys
import shutil
import warnings
from configargparse import ArgumentParser
from pathlib import Path

import database as db
from models.video_preprocessor import VideoPreprocessor
from models.utils import Logger


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, is_config_file=True, help="Specify a config file path")
    parser.add_argument("--base_path", type=str, help="Base path to the dataset")
    parser.add_argument("--db_path", type=str, help="Path to database file")
    parser.add_argument("--max_download", type=int, default=None, help="Max number of videos to download")
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--version", type=str, default=None, help="Version of the dataset")
    parser.add_argument(
        "--num_processes", type=int, default=10,
        help="Number of processes used in multiprocess downloading"
    )
    parser.add_argument(
        "--shot_detector_threshold", type=float, default=27,
        help="Average change in pixel intensity threshold to cut video shot changes"
    )
    parser.add_argument(
        "--clip_filter_threshold", type=float, default=25,
        help="CLIPScore threshold (0-100) to filter video clip with no animal present"
    )
    parser.add_argument(
        "--min_scene_len", type=float, default=50,
        help="Minimum length of a clip"
    )
    parser.add_argument(
        "--max_scene_len", type=float, default=1000,
        help="Maximum length of a clip"
    )
    parser.add_argument(
        "--still_frame_filter_threshold", type=float, default=0.1,
        help="Absolute pixel difference threshold (0-1) to filter clips with still frames"
    )
    parser.add_argument(
        "--keep_discarded", action="store_true", help="Keep discarded clips in a discarded folder, otherwise remove"
    )
    parser.add_argument(
        "--categories", type=str, nargs="+", default=None, help="List of categories to process"
    )
    parsed_args, _ = parser.parse_known_args()
    return parsed_args


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    args = parse_args()
    clip_version = db.parse_version(args.version).get("clip")
    clip_dir = f"clip_{clip_version}" if clip_version else "clip"
    sys.stdout = Logger(Path(args.base_path) / clip_dir / "log.txt")
    preprocessor = VideoPreprocessor(
        db_path=args.db_path,
        output_dir=Path(args.base_path) / clip_dir,
        clip_filter_threshold=args.clip_filter_threshold,
        shot_detector_threshold=args.shot_detector_threshold,
        min_scene_len=args.min_scene_len,
        still_frame_filter_threshold=args.still_frame_filter_threshold,
        keep_discarded=args.keep_discarded,
        version=args.version,
        categories=args.categories
    )
    preprocessor.run()

