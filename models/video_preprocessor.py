import os
import cv2
import torch
import shutil
import random
import logging
import traceback
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from random import shuffle
from time import time
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from torchmetrics.multimodal.clip_score import CLIPScore
from scenedetect import detect, split_video_ffmpeg, ContentDetector, HashDetector, platform
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_detector import FlashFilter

from models.utils import Profiler, timecode_to_second
from database import Database, Status

matplotlib.use("agg")
# platform.init_logger(log_level=logging.DEBUG, show_stdout=True, log_file=None)


class VideoPreprocessor:
    def __init__(
        self, db_path, output_dir, shot_detector_threshold=25, clip_filter_threshold=25, min_scene_len=30,
        max_scene_len=1000, clip_random_sample=10, still_frame_filter_threshold=15, verbose=True, version=None,
        keep_discarded=False, keep_source=False, categories=None,
        detector_weights=ContentDetector.Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=1.0)
    ):
        self.verbose = verbose
        self.keep_discarded = keep_discarded
        self.keep_source = keep_source
        self.categories = categories
        self.output_dir = output_dir
        self.clip_filter_threshold = clip_filter_threshold
        self.clip_random_sample = clip_random_sample
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(self.device)
        self.shot_detector_threshold = shot_detector_threshold
        self.min_scene_len = min_scene_len
        self.max_scene_len = max_scene_len
        self.detector_weights = detector_weights
        self.still_frame_filter_threshold = still_frame_filter_threshold
        self.detector = ContentDetector(
            threshold=shot_detector_threshold, min_scene_len=min_scene_len, weights=detector_weights
        )
        self.init_database(db_path=db_path, version=version)

    def init_database(self, db_path, version):
        self.db = Database(db_path=db_path, version=version)
        self.db.make_clip_table()
        self.db.make_track_table()

    @staticmethod
    def save_stats_plot(scene_list, stats_file_path, plot_path, video_id, threshold):
        """Plot and save scores vs. time from stats file generated when splitting video into clips."""
        df = pd.read_csv(stats_file_path)
        df["Time"] = df["Timecode"].apply(timecode_to_second)
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.plot(df["Time"], df["content_val"], marker='.', linestyle='-', color='b')
        plt.axhline(y=threshold, color='r', linestyle="--", label=f"threshold={threshold}")
        for scene in scene_list:
            start, end = scene
            plt.axvline(x=timecode_to_second(start), color='g', linestyle="--")
            plt.axvline(x=timecode_to_second(end), color='g', linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("content_val")
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.title(video_id)
        plt.savefig(plot_path)
        plt.close()

    def reset_detector(self, min_scene_len):
        """Reset detector state and min_scene_len of new video"""
        if hasattr(self.detector, "_last_frame"):
            self.detector._last_frame = None
        if hasattr(self.detector, "_flash_filter"):
            self.detector._flash_filter = FlashFilter(mode=FlashFilter.Mode.MERGE, length=min_scene_len)
            assert self.detector._flash_filter._last_above is None
            assert self.detector._flash_filter._merge_enabled is False
            assert self.detector._flash_filter._merge_triggered is False
            assert self.detector._flash_filter._merge_start is None

    def split_long_scene(self, scene_list):
        """Recursively split long scenes into halves if exceeds self.max_scene_len"""
        new_scene_list = []
        def split_interval(interval):
            start, end = interval
            if end.frame_num - start.frame_num <= self.max_scene_len:
                new_scene_list.append(interval)
            else:
                mid_frame = int((start.frame_num + end.frame_num) // 2)
                mid = FrameTimecode(mid_frame, fps=start.framerate)
                split_interval((start, mid))
                split_interval((mid, end))
            return
        for scene in scene_list:
            split_interval(scene)
        for scene in new_scene_list:
            start, end = scene
            assert end.frame_num - start.frame_num <= self.max_scene_len
        return new_scene_list

    def split_shot_change(self, video_id, video_path, output_dir):
        """
        Split video into clips <video_id>_<clip_id>.mp4
        TODO: handle ffmpeg error
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if self.verbose:
            print(f"Splitting {video_id} with {total_frames} frames")
        self.reset_detector(min_scene_len=self.min_scene_len)
        stats_file_path = video_path.parent / f"{video_id}_stats.csv"
        assert os.path.exists(video_path), f"{video_path} not found"
        scene_list = detect(str(video_path), self.detector, stats_file_path=str(stats_file_path))
        if len(scene_list) == 0:
            scene_list = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]
        scene_list = self.split_long_scene(scene_list)
        self.save_stats_plot(
            video_id=video_id, stats_file_path=stats_file_path, threshold=self.detector._threshold,
            plot_path=video_path.parent/f"{video_id}_stats.png", scene_list=scene_list
        )
        retval = split_video_ffmpeg(
            str(video_path), scene_list=scene_list, output_dir=output_dir, show_progress=True,
            output_file_template="${VIDEO_NAME}_${SCENE_NUMBER}.mp4"
        )
        clip_paths = sorted([f for f in output_dir.iterdir() if f.is_file() and f.suffix == ".mp4"])
        assert retval == 0, f"FFmpeg returned {retval}"
        assert len(clip_paths) == len(scene_list), f"Split error for {video_id}"
        for clip_path, scene in zip(clip_paths, scene_list):
            clip_path = Path(clip_path)
            duration = scene[1] - scene[0]
            self.db.insert_clip(
                clip_id=clip_path.stem, video_id=clip_path.parent.stem, clip_path=clip_path,
                start_time=timecode_to_second(scene[0]), start_frame=scene[0].frame_num,
                duration=timecode_to_second(duration), frames=duration.frame_num
            )
        if self.verbose:
            print(f"Split {video_id}: {len(clip_paths)} clips", flush=True)
        return clip_paths

    def filter_short_clips(self, clip_paths):
        """
        Input list of paths to video clips
        Remove clips with length < self.min_scene_len
        Return list of paths to filtered clips
        TODO: check update path to NULL
        """
        filtered_paths = []
        for clip_path in tqdm(clip_paths, desc="Filtering short clips"):
            clip_path = Path(clip_path)
            clip_id = clip_path.stem
            cap = cv2.VideoCapture(str(clip_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.min_scene_len:
                filter_reason = f"total_frames ({total_frames}) < self.min_scene_len ({self.min_scene_len})"
                self.db.update_clip(clip_id=clip_id, status=Status.DISCARDED, filter_reason=filter_reason)
                if self.keep_discarded:
                    self.save_discarded_clips(clip_path, filter_reason)
                else:
                    subprocess.run(["rm", "-rf", clip_path])
                    self.db.update_clip(clip_id, clip_path=None)
                if self.verbose:
                    print(f"remove {clip_id}: {filter_reason}", flush=True)
            else:
                filtered_paths.append(clip_path)
        return filtered_paths

    def filter_still_frames(self, clip_paths):
        """
        Input list of paths to video clips
        Remove clips with length < self.min_scene_len
        Remove clips with little or no motion by comparing absolute pixel difference between 10 pairs of random frames
        Return list of paths to filtered clips
        """
        filtered_paths = []
        for clip_path in tqdm(clip_paths, desc="Filtering still frames"):
            clip_path = Path(clip_path)
            clip_id = clip_path.stem
            cap = cv2.VideoCapture(str(clip_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            random_frames = []
            # random_frame_ids = np.random.choice(total_frames, 20, replace=False)  # random frames
            assert total_frames > 20, f"total_frames ({total_frames}) <= 20"
            random_frame_id = np.random.choice(total_frames-20, 1, replace=False)[0]  # consecutive frames
            random_frame_ids = [random_frame_id+i for i in range(20)]  # consecutive frames
            for frame_index in random_frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                random_frames.append(frame)
            cap.release()
            video_array1 = np.stack(random_frames[0::2])
            video_array2 = np.stack(random_frames[1::2])
            assert len(video_array1) == len(video_array2), f"{len(video_array1)} != {len(video_array2)}"
            video_tensor1 = torch.from_numpy(video_array1).float().to(self.device)
            video_tensor2 = torch.from_numpy(video_array2).float().to(self.device)
            video_tensor1 = resize(video_tensor1, [128, 128], antialias=True)
            video_tensor2 = resize(video_tensor2, [128, 128], antialias=True)
            score = (video_tensor1 - video_tensor2).abs().mean().item()
            if score < self.still_frame_filter_threshold:
                filter_reason = (
                    f"avg abs pixel diff ({score}) < self.still_frame_filter_threshold "
                    f"({self.still_frame_filter_threshold})"
                )
                self.db.update_clip(clip_id=clip_id, status=Status.DISCARDED,filter_reason=filter_reason)
                if self.keep_discarded:
                    self.save_discarded_clips(clip_path, filter_reason)
                else:
                    subprocess.run(["rm", "-rf", clip_path])
                    self.db.update_clip(clip_id, clip_path=None)
                if self.verbose:
                    print(f"remove {clip_id}: {filter_reason}", flush=True)
                continue
            filtered_paths.append(clip_path)
        return filtered_paths

    def filter_multi_shots(self, clip_paths):
        """
        Input list of paths to video clips
        Remove clips with multiple shots
        Return list of paths to filtered clips
        """
        filtered_paths = []
        for clip_path in tqdm(clip_paths, desc="Filtering multi shots"):
            self.reset_detector(min_scene_len=2)
            clip_path = Path(clip_path)
            clip_id = clip_path.stem
            scene_list = detect(str(clip_path), self.detector)
            num_shot_changes = len(scene_list) - 1
            if num_shot_changes > 0:
                filter_reason = f"num_shot_changes = {num_shot_changes}"
                self.db.update_clip(clip_id=clip_id, status=Status.DISCARDED, filter_reason=filter_reason)
                if self.keep_discarded:
                    self.save_discarded_clips(clip_path, filter_reason)
                else:
                    subprocess.run(["rm", "-rf", clip_path])
                    self.db.update_clip(clip_id, clip_path=None)
                if self.verbose:
                    print(f"remove {clip_id}: {filter_reason}", flush=True)
            else:
                filtered_paths.append(clip_path)
        return filtered_paths

    def clip_score(self, clip_path, category):
        """
        Input path to a clip
        Return CLIPScore of video with respect to category name
        """
        start = time()
        vidcap = cv2.VideoCapture(str(clip_path))
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_sample = min(total_frames, self.clip_random_sample)
        random_frames = random.sample(range(total_frames), random_sample)
        images = []
        for frame in random_frames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = vidcap.read()
            if success:
                images.append(image)
        if len(images) == 0:
            return 100
        images = np.stack(images)
        images = rearrange(images, "b h w c -> b c h w")
        images = torch.from_numpy(images)
        text = [f"a photo of {category}"] * images.shape[0]
        images = images.to(self.device)
        score = self.metric(images, text).detach().item()
        elapsed = time() - start
        if self.verbose:
            print(f"{elapsed:.2f}s/{total_frames}frames {elapsed/total_frames:.2f}s/frame", flush=True)
        return score

    def filter_clipscore(self, clip_paths, category):
        """
        Input list of paths to video clips
        Remove clips with CLIPScore lower than self.clip_filter_threshold
        Return list of paths to filtered clips
        """
        filtered_paths = []
        for clip_path in tqdm(clip_paths, desc="CLIP filtering"):
            clip_path = Path(clip_path)
            clip_id = clip_path.stem
            score = self.clip_score(clip_path, category)
            if score < self.clip_filter_threshold:
                filter_reason = f"CLIPScore ({score}) < self.clip_filter_threshold ({self.clip_filter_threshold})"
                self.db.update_clip(clip_id=clip_id, status=Status.DISCARDED, filter_reason=filter_reason)
                if self.keep_discarded:
                    self.save_discarded_clips(clip_path, filter_reason)
                else:
                    subprocess.run(["rm", "-rf", clip_path])
                    self.db.update_clip(clip_id, clip_path=None)
                if self.verbose:
                    print(f"remove {clip_id}: {filter_reason}", flush=True)
            else:
                filtered_paths.append(clip_path)
        return filtered_paths

    def check_video_status(self, video_id):
        status = self.db.get_video_status(video_id)
        if status is None:
            print(f"Skip processing {video_id} not in db", flush=True)
            return False
        elif status != Status.DOWNLOADED:
            print(f"Skip processing {video_id} with status {status}", flush=True)
            return False
        else:
            self.db.update_video_status(video_id, Status.PROCESSING)
            self.db.remove_clip(video_id)
            self.db.remove_track(video_id)
            print(f"Processing {video_id}", flush=True)
            return True

    def reset_video_status(self, video_id):
        self.db.update_video(video_id, status=Status.DOWNLOADED)
        self.db.remove_clip(video_id)

    def save_discarded_clips(self, clip_path, filter_reason):
        """Move discarded clips to a discarded folder and write filter reason to a txt file"""
        clip_id = Path(clip_path).stem
        discarded_dir = Path(clip_path).parent / "discarded"
        discarded_path = discarded_dir / clip_path.name
        os.makedirs(discarded_dir, exist_ok=True)
        subprocess.run(["rm", "-rf", discarded_path])
        shutil.move(clip_path, discarded_path)
        with open(discarded_dir / f"{clip_id}.txt", 'w') as f:
            f.write(filter_reason)
        self.db.update_clip(clip_id, clip_path=str(discarded_dir/clip_path.name))

    def update_db_status(self, video_id, clip_paths):
        """Update video status to PROCESSED assert all clip status are DOWNLOADED after processing a video"""
        self.db.update_video_status(video_id, Status.PROCESSED)
        for clip_path in clip_paths:
            clip_id = Path(clip_path).stem
            assert self.db.get_clip_status(clip_id) == Status.DOWNLOADED

    def run(self):
        all_videos = self.db.get_all_videos()
        if self.categories:
            all_videos = [v for v in all_videos if v.get("category") in self.categories]
        shuffle(all_videos)
        for video in tqdm(all_videos):
            video_path = video.get("video_path")
            video_id = video.get("video_id")
            category = video.get("category")
            output_dir = Path(self.output_dir) / category / video_id
            try:
                if not self.check_video_status(video_id):
                    continue
                # with Profiler("split_shot_change", video_ids=[video_id], db=self.db):
                clip_paths = self.split_shot_change(video_id=video_id, video_path=video_path, output_dir=output_dir)
                # with Profiler("filter_short_clips", clip_paths=clip_paths, db=self.db):
                clip_paths = self.filter_short_clips(clip_paths)
                # with Profiler("filter_still_frames", clip_paths=clip_paths, db=self.db):
                clip_paths = self.filter_still_frames(clip_paths)
                # with Profiler("filter_multi_shots", clip_paths=clip_paths, db=self.db):
                clip_paths = self.filter_multi_shots(clip_paths)
                # with Profiler("filter_clipscore", clip_paths=clip_paths, db=self.db):
                clip_paths = self.filter_clipscore(clip_paths, category)
                self.update_db_status(video_id=video_id, clip_paths=clip_paths)
                if not self.keep_source:
                    subprocess.run(["rm", "-rf", video_path])
                    self.db.update_video(video_id, video_path=None)
            except Exception as e:
                print(f"Error processing {video_path}: {type(e).__name__}, {e}", flush=True)
                traceback.print_exc()
                self.reset_video_status(video_id)


# if __name__ == "__main__":
#     processor = VideoPreprocessor(category="horse", shot_detector_threshold=45, db_path="data/database.sqlite")
#     paths = glob('data/stage1/horse/*/*.mp4')
#     processor.run(paths)
