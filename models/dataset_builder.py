import os
import orjson
import json
import random
import torch
import shutil
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import Database, Status


class DatasetBuilder:
    def __init__(
        self, dataset_dir, db_path, version, train_proportion, file_suffixes, filter_method,
        occlusion_filter_threshold, flow_filter_threshold, min_scene_len, seed, verbose=False,
        filter_video_ids=None, filter_clip_ids=None, filter_track_ids=None, keep_track_ids=None
    ):
        self.dataset_dir = dataset_dir
        self.min_scene_len = min_scene_len
        self.file_suffixes = file_suffixes
        self.train_proportion = train_proportion
        self.occlusion_filter_threshold = occlusion_filter_threshold
        self.filter_method = filter_method
        self.flow_filter_threshold = flow_filter_threshold
        self.verbose = verbose
        self.overwrite = False
        self.filter_video_ids = filter_video_ids
        self.filter_clip_ids = filter_clip_ids
        self.filter_track_ids = filter_track_ids
        self.keep_track_ids = keep_track_ids
        self.db = Database(db_path, version=version)
        if seed is not None:
            random.seed(seed)

    def filter_tracks(self, track):
        """Filter track based on average occlusion and flow of the track"""
        video_id = track.get("video_id")
        clip_id = track.get("clip_id")
        track_id = track.get("track_id")
        if self.filter_video_ids and video_id in self.filter_video_ids:
            return []
        if self.filter_clip_ids and clip_id in self.filter_clip_ids:
            return []
        if self.filter_track_ids and track_id in self.filter_track_ids:
            return []
        if self.keep_track_ids and track_id not in self.keep_track_ids:
            return []
        track_path = track.get("track_path")
        track_occlusion = track.get("occlusion")
        track_flow = track.get("flow")
        assert track_occlusion is not None, f"Track {track_id} has no occlusion"
        if track_occlusion is None or track_occlusion == "nan" or (
            self.occlusion_filter_threshold is not None and track_occlusion > self.occlusion_filter_threshold
        ):
            return []
        if track_flow is None or track_flow == "nan" or (
            self.flow_filter_threshold is not None and track_flow < self.flow_filter_threshold
        ):
            return []
        frame_ids = set()
        for filename in os.listdir(track_path):
            if not filename.endswith(".png"):
                continue
            frame_id = int(filename.split('_')[0])
            frame_ids.add(frame_id)
        return sorted(list(frame_ids))

    def filter_frames(self, track):
        """  TODO slow, not fully tested
        Filter frames based on occlusion of the frame in metadata.json
        Keep the longest capable track greater than min_scene_len
        """
        track_path = Path(track.get("track_path"))
        all_frame_ids = set()
        def process_metadata(filename):
            if not filename.name.endswith("metadata.json"):
                return None
            try:
                with open(filename, "rb") as f:
                    data = orjson.loads(f.read())
                occlusion = data.get("occlusion_proportion")
                flow = data.get("flow")
                if occlusion is None or occlusion == "nan" or (
                    self.occlusion_filter_threshold is not None and occlusion > self.occlusion_filter_threshold
                ):
                    return None
                if flow is None or flow == "nan" or (
                    self.flow_filter_threshold is not None and flow < self.flow_filter_threshold
                ):
                    return None
                return int(filename.stem.split('_')[0])
            except Exception as e:
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                traceback.print_exc()
                return None
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_file = {executor.submit(process_metadata, file): file for file in track_path.iterdir()}
            for future in as_completed(future_to_file):
                frame_id = future.result()
                if frame_id is not None:
                    all_frame_ids.add(frame_id)
        frame_ids = []
        sorted_frame_ids = sorted(all_frame_ids)
        current_sequence = []
        for frame_id in sorted_frame_ids:
            if not current_sequence or frame_id == current_sequence[-1] + 1:
                current_sequence.append(frame_id)
            else:
                if len(current_sequence) > len(frame_ids):
                    frame_ids = current_sequence
                current_sequence = [frame_id]
        if len(current_sequence) > len(frame_ids):
            frame_ids = current_sequence
        if len(frame_ids) < self.min_scene_len:
            return []
        return frame_ids

    def copy_data_frame(self, track_id, frame_ids, split, symlink=True):
        if not frame_ids:
            return
        track_info = self.db.get_track(track_id)
        track_path = track_info.get("track_path")
        video_info = self.db.get_video(track_info.get("video_id"))
        category = video_info.get("category")
        dst_dir = os.path.join(self.dataset_dir, split, category, track_id)
        os.makedirs(dst_dir, exist_ok=True)
        frame_ids = [str(fid).zfill(8) for fid in frame_ids]
        files_to_copy = []
        for frame_id in frame_ids:
            for suffix in self.file_suffixes:
                filename = f"{frame_id}_{suffix}"
                files_to_copy.append(filename)
        additional_files = [f"{track_id}_rgb.mp4", f"{track_id}_rgb_masked.mp4"]
        files_to_copy.extend(additional_files)
        if symlink:  # Create symbolic links instead of copy
            for filename in files_to_copy:
                try:
                    src_path = os.path.realpath(os.path.join(track_path, filename))
                    dst_path = os.path.join(dst_dir, filename)
                    os.symlink(src_path, dst_path)
                except Exception as e:
                    shutil.rmtree(dst_dir, ignore_errors=True)
                    print(f"{type(e).__name__}: {str(e)}", flush=True)
                    traceback.print_exc()
        else:  # Copy files
            file_list_path = os.path.join(dst_dir, "tmp_file_list.txt")
            with open(file_list_path, 'w') as file_list:
                for filename in files_to_copy:
                    file_list.write(f"{filename}\n")
            rsync_command = [
                "rsync", "-a", "--files-from", file_list_path, track_path.rstrip('/') + '/', dst_dir.rstrip('/')
            ]
            try:
                subprocess.run(rsync_command, check=True)
            except Exception as e:
                shutil.rmtree(dst_dir, ignore_errors=True)
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                traceback.print_exc()
            finally:
                os.remove(file_list_path)

    def copy_data_track(self, track_id, frame_ids, split, symlink=True):
        if not frame_ids:
            return
        track_info = self.db.get_track(track_id)
        track_path = track_info.get("track_path")
        video_info = self.db.get_video(track_info.get("video_id"))
        category = video_info.get("category")
        dst_dir = os.path.join(self.dataset_dir, split, category)
        dst_track_dir = os.path.join(dst_dir, track_id)
        os.makedirs(dst_dir, exist_ok=True)
        if symlink:  # Create symbolic links instead of copy
            try:
                os.symlink(os.path.realpath(track_path), dst_track_dir)
            except Exception as e:
                shutil.rmtree(dst_track_dir, ignore_errors=True)
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                traceback.print_exc()
        else:  # Copy files
            rsync_command = ["rsync", "-az", track_path.rstrip('/'), dst_dir.rstrip('/')]
            try:
                subprocess.run(rsync_command, check=True)
            except Exception as e:
                shutil.rmtree(dst_track_dir, ignore_errors=True)
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                traceback.print_exc()

    def run(self):
        all_tracks = self.db.get_all_tracks()
        all_tracks = [t for t in all_tracks if t.get("location") == "viscam" and t.get("gpt_filtered") != False]
        all_track_ids = [t.get("track_id") for t in all_tracks]
        print(f"Total initial tracks: {len(all_tracks)}", flush=True)
        print(f"Filter method: {self.filter_method}", flush=True)
        random.shuffle(all_tracks)
        track_id_to_frames = {}
        for track in tqdm(all_tracks):
            try:
                if self.filter_method is None or self.filter_method == "None":
                    track_path = track.get("track_path")
                    frame_ids = set()
                    for filename in os.listdir(track_path):
                        if not filename.endswith(".png"):
                            continue
                        frame_id = int(filename.split('_')[0])
                        frame_ids.add(frame_id)
                    frame_ids = sorted(list(frame_ids))
                elif self.filter_method == "track":
                    frame_ids = self.filter_tracks(track)
                elif self.filter_method == "frame":
                    frame_ids = self.filter_frames(track)
                else:
                    raise NotImplementedError
            except Exception as e:
                print(f"Error processing track {track.get('track_id')}: {str(e)}", flush=True)
                traceback.print_exc()
                exit()
                continue
            track_id_to_frames[track.get("track_id")] = frame_ids
        total_frames = sum([len(v) for v in track_id_to_frames.values()])
        total_tracks = sum([1 for v in track_id_to_frames.values() if len(v) > 0])
        if self.verbose:
            print(f"Total filtered frames: {total_frames}", flush=True)
            print(f"Total filtered tracks: {total_tracks}", flush=True)
        train_frames = test_frames = 0
        train_tracks = test_tracks = 0
        with tqdm(total=self.train_proportion * total_frames, desc="Copy train split", disable=not self.verbose) as pbar:
            while train_frames < self.train_proportion * total_frames:
                track_id = all_track_ids.pop()
                frame_ids = track_id_to_frames.pop(track_id)
                train_frames += len(frame_ids)
                train_tracks += 1 if len(frame_ids) > 0 else 0
                pbar.update(len(frame_ids))
                if self.filter_method == "frame":
                    self.copy_data_frame(track_id, frame_ids, split="train")
                elif self.filter_method == "track":
                    self.copy_data_track(track_id, frame_ids, split="train")
                elif self.filter_method is None or self.filter_method == "None":
                    self.copy_data_track(track_id, frame_ids, split="train")
                else:
                    raise NotImplementedError
        if self.verbose:
            print(f"Train frames: {train_frames}", flush=True)
            print(f"Train tracks: {train_tracks}", flush=True)
        with tqdm(total=total_frames - train_frames, desc="Copy test split", disable=not self.verbose) as pbar:
            for track_id in all_track_ids:
                frame_ids = track_id_to_frames.pop(track_id)
                test_frames += len(frame_ids)
                test_tracks += 1 if len(frame_ids) > 0 else 0
                pbar.update(len(frame_ids))
                if self.filter_method == "frame":
                    self.copy_data_frame(track_id, frame_ids, split="test")
                elif self.filter_method == "track":
                    self.copy_data_track(track_id, frame_ids, split="test")
                elif self.filter_method is None or self.filter_method == "None":
                    self.copy_data_track(track_id, frame_ids, split="test")
                else:
                    raise NotImplementedError
        if self.verbose:
            print(f"Test frames: {test_frames}", flush=True)
            print(f"Test tracks: {test_tracks}", flush=True)
