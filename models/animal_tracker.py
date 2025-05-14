import json
import os
import random
from time import sleep
import cv2
import math
import traceback
import shutil
from tqdm import tqdm
from glob import glob
import numpy as np
from pathlib import Path
import torch.cuda
from PIL import Image
import supervision as sv
from copy import deepcopy
from random import shuffle
from moviepy.editor import VideoFileClip
from openai import OpenAI
from models.occlusion_processor import OcclusionProcessor
from models.flow_processor import FlowProcessor
from models.pose_processor import PoseProcessor
from models.trackers import GroundedSAM2Tracker
from models.utils import get_frame_images, Profiler, crop, gpt_filter
from database import Database, Status
from tools.copy_results import copy_results


class AnimalTracker:  # TODO: maybe rename to object tracker
    def __init__(
        self, output_dir, tracking_method, db_path, compute_occlusion, compute_flow, overlap_iou_threshold,
        truncation_border_width, depth_method, grounding_text_format, bbox_area_threshold, sam2_prompt_type,
        crop_height, crop_width, fps, crop_margin, device, tracking_sam_step, consistency_iou_threshold, flow_method,
        flow_batch_size, compute_pose, pose_method, pose_batch_size, min_scene_len, occlusion_batch_size, verbose,
        save_visualization, max_track_gap, save_local_dir, version, keep_source=False, categories=None
    ):
        self.output_dir = output_dir
        self.verbose = verbose
        self.device = device
        self.compute_occlusion = compute_occlusion
        self.compute_flow = compute_flow
        self.compute_pose = compute_pose
        self.keep_source = keep_source
        self.categories = categories
        self.tracking_sam_step = tracking_sam_step
        self.overlap_iou_threshold = overlap_iou_threshold
        self.consistency_iou_threshold = consistency_iou_threshold
        self.truncation_border_width = truncation_border_width
        self.grounding_text_format = grounding_text_format
        self.bbox_area_threshold = bbox_area_threshold
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_margin = crop_margin
        self.min_scene_len = min_scene_len
        self.sam2_prompt_type = sam2_prompt_type
        self.max_track_gap = max_track_gap
        self.fps = fps
        self.save_visualization = save_visualization
        self.save_local_dir = save_local_dir
        self.gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
        self.image_suffix = "rgb.png"
        self.masked_image_suffix = "rgb_masked.png"
        self.mask_suffix = "mask.png"
        self.metadata_suffix = "metadata.json"
        self.occlusion_suffix = "occlusion.png"
        self.flow_suffix = "flow.png"
        self.depth_suffix = "depth.png"
        self.keypoint_suffix = "keypoint.txt"
        self.pose_image_suffix = "pose.png"
        self.init_database(version=version, db_path=db_path)

        with Profiler("init_tracker"):
            if tracking_method == "naive":
                # TODO: add args for naive tracker and samtrack
                # self.tracker = NaiveTracker(args)
                raise NotImplementedError
            elif tracking_method == "sam_track":
                # TODO: add samtrack implementation
                raise NotImplementedError
                # self.tracker = SAMTracker()
            elif tracking_method == "grounded_sam2":
                self.tracker = GroundedSAM2Tracker(
                    sam2_prompt_type=self.sam2_prompt_type,
                    tracking_sam_step=self.tracking_sam_step,
                    device=self.device
                )
            else:
                raise NotImplementedError
        if self.compute_occlusion:
            with Profiler("init_occlusion_processor"):
                self.occlusion_processor = OcclusionProcessor(
                    depth_method=depth_method,
                    occlusion_batch_size=occlusion_batch_size,
                    occlusion_suffix=self.occlusion_suffix,
                    depth_suffix=self.depth_suffix,
                    mask_suffix=self.mask_suffix,
                    device=self.device
                )
        if self.compute_flow:
            with Profiler("init_flow_estimator"):
                self.flow_processor = FlowProcessor(
                    flow_method=flow_method,
                    image_suffix=self.image_suffix,
                    flow_suffix=self.flow_suffix,
                    mask_suffix=self.mask_suffix,
                    flow_batch_size=flow_batch_size,
                    device=self.device
                )
        if self.compute_pose:
            with Profiler("init_pose_processor"):
                self.pose_processor = PoseProcessor(
                    pose_method=pose_method,
                    image_suffix=self.image_suffix,
                    keypoint_suffix=self.keypoint_suffix,
                    pose_image_suffix=self.pose_image_suffix,
                    batch_size=pose_batch_size,
                    device=self.device,
                    save_pose_image=True
                )

    def init_database(self, db_path, version):
        self.db = Database(db_path=db_path, version=version)
        self.db.make_track_table()

    def extract_frames(self, clip_path, save_dir):
        """
        Extract frames png in the specified directory.
        Resample frames to self.fps if not None.
        """
        os.makedirs(save_dir, exist_ok=True)
        clip = VideoFileClip(clip_path)
        original_fps = clip.fps
        self.fps = self.fps if self.fps is not None else original_fps
        if self.verbose and original_fps != self.fps:
            print(f"Resampling {clip_path} from {original_fps} to {self.fps} fps", flush=True)
        saved_frame_count = 0
        for frame_idx, (frame_time, frame) in enumerate(clip.iter_frames(fps=self.fps, with_times=True, dtype="uint8")):
            frame_filename = os.path.join(save_dir, f"{str(saved_frame_count).zfill(8)}.png")
            cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_frame_count += 1
        clip.reader.close()

    def track(self, frames_dir, category):
        grounding_text_input = self.grounding_text_format.format(category) if self.grounding_text_format else category
        tracking_results, grounding_scores = self.tracker(frames_dir=frames_dir, grounding_text_input=grounding_text_input)
        return tracking_results, grounding_scores

    def check_clip_status(self, clip_path):
        clip_id = Path(clip_path).stem
        status = self.db.get_clip_status(clip_id)
        if status is None:
            print(f"Skip processing {clip_id} not in db", flush=True)
            return False
        elif status != Status.DOWNLOADED:
            print(f"Skip processing {clip_id} with status {status}", flush=True)
            return False
        else:
            self.db.update_clip(clip_id, status=Status.PROCESSING)
            self.db.remove_track(clip_id=clip_id)
            print(f"Processing {clip_id}", flush=True)
            return True

    def reset_clip_status(self, clip_id):
        self.db.update_clip(clip_id, status=Status.DOWNLOADED)
        self.db.remove_track(clip_id=clip_id)

    def update_db_status(self, clip_id, clip_path):
        if not self.keep_source:
            try:
                os.remove(clip_path)
            except:
                pass
            self.db.update_clip(clip_id, status=Status.PROCESSED, clip_path=None)
        else:
            self.db.update_clip(clip_id, status=Status.PROCESSED)

    @staticmethod
    def get_tracked_frame_ids(tracking_results):
        """Get all frame ids with tracking results"""
        frame_ids = []
        for frame_id, segments in tracking_results.items():
            if len(segments) > 0:
                for object_id, mask in segments.items():
                    if np.any(mask):
                        frame_ids.append(frame_id)
                        break
        return frame_ids

    @staticmethod
    def get_tracked_object_ids(tracking_results):
        """Get all object ids in tracking results"""
        object_ids = set()
        for _, segments in tracking_results.items():
            for object_id, _ in segments.items():
                object_ids.add(object_id)
        return list(object_ids)

    @staticmethod
    def clean_tracking_results(tracking_results):
        """Remove masks in tracking results with no detection"""
        new_tracking_results = {}
        for frame_id, segments in tracking_results.items():
            new_tracking_results[frame_id] = {}
            for object_id, mask in segments.items():
                if mask.any():
                    new_tracking_results[frame_id][object_id] = mask
        return new_tracking_results

    def split_discontinuous_track(self, tracking_results, grounding_scores=None):
        """
        Split tracks with potential incontinuity after removing detection in tracking results
        When a large gap exists for an object, the previous track segment is:
        1) Removed if it is shorter than self.min_scene_len
        2) Kept intact if it is the first track segment
        3) Add to tracking results with a new object id if it is not the first track segment
        """
        new_tracking_results = {}
        all_object_ids = sorted(self.get_tracked_object_ids(tracking_results))
        all_object_ids_to_iterate = deepcopy(all_object_ids)
        for object_id in all_object_ids_to_iterate:
            all_presented_frames = []
            for frame in tracking_results.keys():
                if object_id in tracking_results[frame].keys():
                    if np.any(tracking_results[frame][object_id]):
                        all_presented_frames.append(frame)
                    else:
                        del tracking_results[frame][object_id]
            all_presented_frames = sorted(all_presented_frames)
            start = None
            end = None
            next_present = None
            last_present = None
            new_track = False
            for idx, current in enumerate(all_presented_frames):
                if start is None:  # first frame
                    start = current
                if idx + 1 == len(all_presented_frames):  # last frame
                    next_present = -1
                    end = current + 1
                else:  # not last frame
                    next_present = all_presented_frames[idx+1]
                    if next_present - current - 1 > self.max_track_gap:  # large gap
                        end = current + 1
                if end is not None:
                    # last frame or large gap between current and next_present , need to decide
                    # 1) remove this split
                    # 2) keep this split as first track
                    # 3) add this split as a new track
                    if end - start < self.min_scene_len:  # 1) remove this split
                        for f in range(start, end):
                            if object_id in tracking_results[f].keys():
                                del tracking_results[f][object_id]
                    elif not new_track:  # 2) keep this split as first track, mark later track as new track
                        new_track = True
                    else:  # 3) add this split as a new track
                        new_object_id = int(np.max(all_object_ids) + 1)
                        all_object_ids.append(new_object_id)
                        for f in range(start, end):
                            if f not in all_presented_frames:
                                continue
                            if f not in new_tracking_results.keys():
                                new_tracking_results[f] = {}
                            new_tracking_results[f][new_object_id] = tracking_results[f][object_id]
                            del tracking_results[f][object_id]
                        if grounding_scores is not None:
                            grounding_scores[new_object_id] = grounding_scores[object_id]
                    start = next_present
                    end = None
        for frame_id, segments in new_tracking_results.items():
            if frame_id not in tracking_results.keys():
                tracking_results[frame_id] = {}
            for object_id, mask in segments.items():
                tracking_results[frame_id][object_id] = mask
        return tracking_results, grounding_scores

    def filter_small_detection(self, tracking_results):
        if len(tracking_results) == 0:
            return tracking_results
        object_ids_to_remove = set()
        for frame_id, segments in tracking_results.items():
            for object_id, mask in segments.items():
                x_min, y_min, x_max, y_max = sv.mask_to_xyxy(mask).squeeze()
                if (x_max - x_min) * (y_max - y_min) < self.bbox_area_threshold:
                    object_ids_to_remove.add((frame_id, object_id))
        for idx in object_ids_to_remove:
            del tracking_results[idx[0]][idx[1]]
        return tracking_results

    @staticmethod
    def mask_bbox_iou(mask1, mask2):
        if not mask1.any() or not mask2.any():
            return 0
        x_min1, y_min1, x_max1, y_max1 = sv.mask_to_xyxy(mask1).squeeze()
        x_min2, y_min2, x_max2, y_max2 = sv.mask_to_xyxy(mask2).squeeze()
        inter_x_min = max(x_min1, x_min2)
        inter_y_min = max(y_min1, y_min2)
        inter_x_max = min(x_max1, x_max2)
        inter_y_max = min(y_max1, y_max2)
        inter_width = max(0, inter_x_max - inter_x_min + 1)
        inter_height = max(0, inter_y_max - inter_y_min + 1)
        inter_area = inter_width * inter_height
        area1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
        area2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    @staticmethod
    def bbox_iou(bbox1, bbox2):
        x_min1, y_min1, x_max1, y_max1 = bbox1
        x_min2, y_min2, x_max2, y_max2 = bbox2
        inter_x_min = max(x_min1, x_min2)
        inter_y_min = max(y_min1, y_min2)
        inter_x_max = min(x_max1, x_max2)
        inter_y_max = min(y_max1, y_max2)
        inter_width = max(0, inter_x_max - inter_x_min + 1)
        inter_height = max(0, inter_y_max - inter_y_min + 1)
        inter_area = inter_width * inter_height
        area1 = (x_max1 - x_min1 + 1) * (y_max1 - y_min1 + 1)
        area2 = (x_max2 - x_min2 + 1) * (y_max2 - y_min2 + 1)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou

    def filter_overlap(self, tracking_results):
        """Remove both masks in the same frame if their bbox IoU is greater than self.overlap_iou_threshold"""
        if len(tracking_results) == 0:
            return tracking_results
        object_ids_to_remove = set()
        for frame_id, segments in tracking_results.items():
            for curr_object_id, curr_mask in segments.items():
                for other_object_id, other_mask in segments.items():
                    if other_object_id == curr_object_id:
                        continue
                    if self.mask_bbox_iou(curr_mask, other_mask) > self.overlap_iou_threshold:
                        object_ids_to_remove.add((frame_id, curr_object_id))
                        object_ids_to_remove.add((frame_id, other_object_id))
        for idx in object_ids_to_remove:
            del tracking_results[idx[0]][idx[1]]
        return tracking_results

    @staticmethod
    def filter_truncation(tracking_results, border_width=1):
        """Remove masks with bbox truncation at the border of the frame"""
        if len(tracking_results) == 0:
            return tracking_results
        object_ids_to_remove = set()
        for frame_id, segments in tracking_results.items():
            for object_id, mask in segments.items():
                height, width = mask.squeeze().shape
                x_min, y_min, x_max, y_max = sv.mask_to_xyxy(mask).squeeze()
                if x_min < border_width or y_min < border_width \
                or x_max >= (width - border_width) or y_max >= (height - border_width):
                    object_ids_to_remove.add((frame_id, object_id))
        for idx in object_ids_to_remove:
            del tracking_results[idx[0]][idx[1]]
        return tracking_results

    def filter_inconsistent_track(self, tracking_results):
        """Remove tracks with inconsistent detection (sudden bbox change) by thresholding iou in adjacent frames"""
        if len(tracking_results) == 0:
            return tracking_results
        all_object_ids = self.get_tracked_object_ids(tracking_results)
        object_ids_to_remove = set()
        for object_id in all_object_ids:
            last_mask = None
            for frame in tracking_results.keys():
                mask = tracking_results[frame].get(object_id)
                if mask is None or not np.any(mask):
                    continue
                if last_mask is None:
                    last_mask = mask
                    continue
                if self.mask_bbox_iou(mask, last_mask) < self.consistency_iou_threshold:
                    object_ids_to_remove.add(object_id)
                    break
                last_mask = mask
        for frame_id, segments in tracking_results.items():
            for idx in object_ids_to_remove:
                if idx in segments.keys():
                    del tracking_results[frame_id][idx]
        return tracking_results

    def filter_short_track(self, tracking_results):
        """Remove tracks with length shorter than self.min_scene_len"""
        all_object_ids = self.get_tracked_object_ids(tracking_results)
        object_ids_to_remove = []
        for object_id in all_object_ids:
            presented_frames = []
            for frame in tracking_results.keys():
                if object_id in tracking_results[frame].keys() and np.any(tracking_results[frame][object_id]):
                    presented_frames.append(frame)
            if len(presented_frames) < self.min_scene_len:
                object_ids_to_remove.append(object_id)
        for frame_id, segments in tracking_results.items():
            for idx in object_ids_to_remove:
                if idx in segments.keys():
                    del tracking_results[frame_id][idx]
        return tracking_results

    def save_video(self, save_path, frames_paths):
        if len(frames_paths) == 0:
            return
        image = Image.open(frames_paths[0])
        width, height = image.width, image.height
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (width, height))
        for rgb_file in frames_paths:
            frame = cv2.imread(rgb_file)
            frame_num = ''.join(c for c in Path(rgb_file).name if c.isdigit())
            position = (10, height - 10)
            cv2.putText(frame, frame_num, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video_writer.write(frame)
        video_writer.release()


    @staticmethod
    def get_tracking_crop_boxes(tracking_results):
        def bbox_to_crop_box(bbox, hw_ratio=1):
            """
            Get the crop box (x_min, y_min, x_max, y_max) of an instance based on its bbox in original image
            Crop box size is based on a given hw_ratio and the bbox area
            """
            x_min, y_min, x_max, y_max = bbox
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            bbox_width, bbox_height = x_max - x_min, y_max - y_min
            bbox_area = bbox_width * bbox_height
            crop_area = 2 * bbox_area
            crop_width = math.sqrt(crop_area / hw_ratio)
            crop_height = crop_width * hw_ratio
            crop_x_min = int(x_center - crop_width / 2)
            crop_y_min = int(y_center - crop_height / 2)
            crop_x_max = int(x_center + crop_width / 2)
            crop_y_max = int(y_center + crop_height / 2)
            return np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max])
        tracking_crop_boxes = {}
        for frame_id, segments in tracking_results.items():
            for object_id, mask in segments.items():
                if np.any(mask):
                    bbox = sv.mask_to_xyxy(mask).squeeze()
                    if frame_id not in tracking_crop_boxes.keys():
                        tracking_crop_boxes[frame_id] = {}
                    tracking_crop_boxes[frame_id][object_id] = bbox_to_crop_box(bbox)
        return tracking_crop_boxes

    @staticmethod
    def get_tracking_bboxes(tracking_results):
        """Get tracking bboxes for each object in each frame"""
        tracking_bboxes = {}
        for frame_id, segments in tracking_results.items():
            tracking_bboxes[frame_id] = {}
            for object_id, mask in segments.items():
                if np.any(mask):
                    tracking_bboxes[frame_id][object_id] = sv.mask_to_xyxy(mask).squeeze()
        return tracking_bboxes

    def smooth_crop_boxes(self, tracking_crop_boxes):
        """
        Smooth tracking crop boxes by averaging the crop boxes in a window of frames
        Window size is 2 * (self.max_track_gap + 1) + 1
        """
        new_tracking_crop_boxes = {}
        all_object_ids = self.get_tracked_object_ids(tracking_crop_boxes)
        for object_id in all_object_ids:
            first_presented_frame = max(tracking_crop_boxes.keys())
            last_presented_frame = 0  # inclusive
            for frame_id in tracking_crop_boxes.keys():
                if object_id in tracking_crop_boxes[frame_id].keys():
                    first_presented_frame = min(first_presented_frame, frame_id)
                    last_presented_frame = max(last_presented_frame, frame_id)
            for frame_id in range(first_presented_frame, last_presented_frame + 1):
                if frame_id < first_presented_frame + self.max_track_gap \
                or frame_id > last_presented_frame - self.max_track_gap:  # do not smooth beginning and end
                    if frame_id in tracking_crop_boxes.keys() and object_id in tracking_crop_boxes[frame_id].keys():
                        if frame_id not in new_tracking_crop_boxes.keys():
                            new_tracking_crop_boxes[frame_id] = {}
                        new_tracking_crop_boxes[frame_id][object_id] = tracking_crop_boxes[frame_id][object_id]
                    continue
                if frame_id not in new_tracking_crop_boxes.keys():
                    new_tracking_crop_boxes[frame_id] = {}
                window_start = int(max(first_presented_frame, frame_id - self.max_track_gap - 1))
                window_end = int(min(frame_id + self.max_track_gap + 1, last_presented_frame))
                presented_frames_in_window = [
                    frame for frame in range(window_start, window_end + 1)
                    if frame in tracking_crop_boxes.keys() and object_id in tracking_crop_boxes[frame].keys()
                ]
                assert len(presented_frames_in_window) > 0
                presented_crop_boxes = np.array(
                    [tracking_crop_boxes[frame][object_id] for frame in presented_frames_in_window]
                )
                average_crop_box = np.mean(presented_crop_boxes, axis=0).astype(int)
                new_tracking_crop_boxes[frame_id][object_id] = average_crop_box
        return new_tracking_crop_boxes

    def fill_missing_masks(self, tracking_results, tracking_crop_boxes, frames_dir):
        """Fill missing masks in tracking_results using sam and interpolated bbox as prompt"""
        all_object_ids = self.get_tracked_object_ids(tracking_crop_boxes)
        for object_id in all_object_ids:
            for frame_id, crop_boxes in tracking_crop_boxes.items():
                if object_id not in crop_boxes.keys():
                    continue
                if object_id in tracking_results[frame_id].keys() and np.any(tracking_results[frame_id][object_id]):
                    continue
                previous_presented_frame = next_presented_frame = frame_id
                while previous_presented_frame >= min(tracking_results.keys()):
                    previous_presented_frame -= 1
                    if previous_presented_frame in tracking_results.keys() \
                    and object_id in tracking_results[previous_presented_frame].keys() \
                    and np.any(tracking_results[previous_presented_frame][object_id]):
                        break
                while next_presented_frame <= max(tracking_results.keys()):
                    next_presented_frame += 1
                    if next_presented_frame in tracking_results.keys() \
                    and object_id in tracking_results[next_presented_frame].keys() \
                    and np.any(tracking_results[next_presented_frame][object_id]):
                        break
                if previous_presented_frame < min(tracking_results.keys()) \
                or next_presented_frame > max(tracking_results.keys()):
                    continue
                previous_bbox = sv.mask_to_xyxy(tracking_results[previous_presented_frame][object_id]).squeeze()
                next_bbox = sv.mask_to_xyxy(tracking_results[next_presented_frame][object_id]).squeeze()
                alpha = (frame_id - previous_presented_frame) / (next_presented_frame - previous_presented_frame)
                interpolated_bbox = [int(x) for x in (1 - alpha) * previous_bbox + alpha * next_bbox]
                margin = 5
                frame_height, frame_width = tracking_results[previous_presented_frame][object_id].squeeze().shape
                interpolated_bbox = np.array([
                    max(interpolated_bbox[0] - margin, 0),
                    max(interpolated_bbox[1] - margin, 0),
                    min(interpolated_bbox[2] + margin, frame_width),
                    min(interpolated_bbox[3] + margin, frame_height)
                ])
                if isinstance(self.tracker, GroundedSAM2Tracker):
                    self.tracker.image_predictor.set_image(
                        np.array(get_frame_images(frames_dir=frames_dir, frame_ids=frame_id)[0])
                    )
                    with torch.autocast(device_type=self.device, dtype=self.tracker.mixed_precision):
                        mask, _, _ = self.tracker.image_predictor.predict(box=interpolated_bbox, multimask_output=False)
                        mask = mask.astype(bool)
                else:
                    # TODO: if using naive tracker or samtrack
                    raise NotImplementedError
                if frame_id not in tracking_results.keys():
                    tracking_results[frame_id] = {}
                tracking_results[frame_id][object_id] = mask
        return tracking_results


    def save_cropped_results(
        self, tracking_masks, tracking_crop_boxes, save_dir, frames_dir=None, depths_dir=None
    ):
        # For flow computing between consecutive frames of same obj
        # Maybe move occlusion calculation to outside to batchify for each track
        # Maybe move save metadata & database outside after occlusion and flow computation
        os.makedirs(save_dir, exist_ok=True)
        clip_id = Path(save_dir).name
        clip_info = self.db.get_clip(clip_id)
        clip_id = clip_info.get("clip_id")
        video_id = clip_info.get("video_id")
        video_info = self.db.get_video(video_id)
        category = video_info.get("category")
        track_ids = []
        all_object_ids = self.get_tracked_object_ids(tracking_crop_boxes)
        for object_id in all_object_ids:
            frame_id_to_occlusion, frame_id_to_flow, frame_id_to_metadata = {}, {}, {}
            track_id = f"{clip_id}_{str(object_id).zfill(3)}"
            track_ids.append(track_id)
            track_dir = save_dir / track_id
            rgb_crops, mask_crops, rgb_masked_crops, depth_crops, save_names = [], [], [], [], []
            os.makedirs(track_dir, exist_ok=True)
            for frame_id, crop_boxes in tracking_crop_boxes.items():
                if object_id not in crop_boxes.keys():
                    continue
                crop_box = crop_boxes[object_id]
                image = np.array(get_frame_images(frames_dir=frames_dir, frame_ids=frame_id)[0])
                mask = tracking_masks[frame_id].get(object_id)
                assert mask is not None, f"Object {object_id} not in frame {frame_id}"
                save_name = track_dir / str(frame_id).zfill(8)
                save_names.append(save_name)
                rgb_crop = crop(image, crop_box, self.crop_height, self.crop_width)
                rgb_crops.append(rgb_crop)
                mask_crop = crop(mask.squeeze().astype(np.uint8), crop_box, self.crop_height, self.crop_width)
                # cv2.imwrite(f"{save_name}_{self.image_suffix}", cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
                rgb_masked_crop = np.stack([mask_crop] * 3, axis=-1) * rgb_crop
                rgb_masked_crops.append(rgb_masked_crop)
                # cv2.imwrite(f"{save_name}_{self.masked_image_suffix}", cv2.cvtColor(rgb_masked_crop, cv2.COLOR_RGB2BGR))
                mask_crop = (mask_crop * 255).astype(np.uint8).squeeze()
                mask_crops.append(mask_crop)
                # cv2.imwrite(f"{save_name}_{self.mask_suffix}", mask_crop)
                if self.compute_occlusion:
                    depth = np.array(get_frame_images(frames_dir=depths_dir, frame_ids=frame_id)[0])
                    depth_crop = crop(depth, crop_box, self.crop_height, self.crop_width)
                    depth_crops.append(depth_crop)
                    # cv2.imwrite(f"{save_name}_{self.depth_suffix}", depth_crop)
                video_frame_id = round(
                    clip_info.get("start_frame") + frame_id*(video_info.get("fps") / self.fps)
                )
                clip_frame_time = frame_id / self.fps
                video_frame_time = clip_info.get("start_time") + clip_frame_time
                frame_id_to_metadata[frame_id] = {
                    "video_frame_id": video_frame_id,
                    "video_frame_time": video_frame_time,
                    "clip_frame_id": frame_id,
                    "clip_frame_time": clip_frame_time,
                    "crop_box_xyxy": [int(x) for x in crop_box.squeeze()],
                    "video_frame_height": mask.shape[1],
                    "video_frame_width": mask.shape[2],
                    "crop_height": self.crop_height,
                    "crop_width": self.crop_width,
                }
            if gpt_filter(gpt_client=self.gpt_client, image=random.choice(rgb_crops), category=category) is False:
                print(f"{track_id} filtered by GPT", flush=True)
                shutil.rmtree(track_dir)
                continue
            assert len(rgb_crops) == len(mask_crops) == len(rgb_masked_crops) == len(save_names)
            if self.compute_occlusion:
                assert len(rgb_crops) == len(depth_crops)
            # Save cropped images
            for idx, save_name in enumerate(save_names):
                cv2.imwrite(f"{save_name}_{self.image_suffix}", cv2.cvtColor(rgb_crops[idx], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{save_name}_{self.masked_image_suffix}", cv2.cvtColor(rgb_masked_crops[idx], cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"{save_name}_{self.mask_suffix}", mask_crops[idx])
                if self.compute_occlusion:
                    cv2.imwrite(f"{save_name}_{self.depth_suffix}", depth_crops[idx])

            if self.compute_occlusion:
                with Profiler("occlusion_processor", num_frames=len(frame_id_to_metadata)):
                    frame_id_to_occlusion = self.occlusion_processor.run_track(track_dir)
                    avg_occlusion = np.mean(list(frame_id_to_occlusion.values()))
                    for frame_id, occlusion in frame_id_to_occlusion.items():
                        frame_id_to_metadata[frame_id]["occlusion"] = occlusion
            else:
                avg_occlusion = None
            if self.compute_flow:
                with Profiler("flow_processor", num_frames=len(frame_id_to_metadata)):
                    frame_id_to_flow = self.flow_processor.run_track(track_dir)
                    avg_flow = np.mean(list(frame_id_to_flow.values()))
                    for frame_id, flow in frame_id_to_flow.items():
                        frame_id_to_metadata[frame_id]["flow"] = flow
            else:
                avg_flow = None
            if self.compute_pose:
                with Profiler("pose_processor", num_frames=len(frame_id_to_metadata)):
                    self.pose_processor.run_track(track_dir)
            # Write metadata
            for frame_id, metadata in frame_id_to_metadata.items():
                save_name = track_dir / str(frame_id).zfill(8)
                with open(f"{save_name}_{self.metadata_suffix}", 'w') as f:
                    json.dump(metadata, f, indent=4)
            # Write track info
            video_info = self.db.get_video(video_id)
            clip_info = self.db.get_clip(clip_id)
            track_info = {
                "video_id": video_id,
                "clip_id": clip_id,
                "track_id": track_id,
                "video_title": video_info.get("title"),
                "video_duration": video_info.get("duration"),
                "video_fps": video_info.get("fps"),
                "fps": self.fps,
                "query_text": video_info.get("query_text"),
                "keywords": video_info.get("keywords"),
                "clip_start_time": clip_info.get("start_time"),
                "clip_start_frame": clip_info.get("start_frame"),
                "duration": clip_info.get("duration"),
            }
            with open(track_dir / f"{track_id}_info.json", 'w') as f:
                json.dump(track_info, f, indent=4)
            if self.save_visualization:
                save_path = track_dir / f"{track_id}_rgb.mp4"
                rgb_files = sorted(glob(os.path.join(track_dir, f"*{self.image_suffix}")))
                self.save_video(save_path, rgb_files)
                save_path = track_dir / f"{track_id}_rgb_masked.mp4"
                rgb_masked_files = sorted(glob(os.path.join(track_dir, f"*{self.masked_image_suffix}")))
                self.save_video(save_path, rgb_masked_files)
            self.db.insert_track(
                track_id=track_id, clip_id=clip_id, video_id=video_id, track_path=track_dir,
                length=len(frame_id_to_metadata), occlusion=avg_occlusion, flow=avg_flow
            )
            if self.verbose:
                print(f"Saving {track_dir}")
        return track_ids

    def save_tracking_visualization(self, tracking_results, frames_dir, save_path, grounding_scores=None):
        """Save overall tracking results of all object in the original video"""
        os.makedirs(save_path.parent, exist_ok=True)
        img = get_frame_images(frames_dir=frames_dir, frame_ids=0)[0]
        width, height = img.width, img.height
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (width, height))
        for frame_idx, segments in tracking_results.items():
            img = np.asarray(get_frame_images(frames_dir=frames_dir, frame_ids=frame_idx)[0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0) if len(masks) > 0 else np.zeros((0, height, width), dtype=bool)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks), mask=masks, class_id=np.array(object_ids, dtype=np.int32)
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            if grounding_scores is not None:
                labels = [f"{str(oid).zfill(3)}({grounding_scores.get(oid):.2f})" for oid in object_ids]
            else:
                labels = [str(oid).zfill(3) for oid in object_ids]
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            position = (10, height - 10)
            cv2.putText(
                annotated_frame, str(frame_idx).zfill(8), position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            video_writer.write(annotated_frame)
        video_writer.release()
        if self.verbose:
            print(f"Tracking visualization saved at {save_path}")

    def run(self, clip_path=None):
        clip_list = self.db.get_all_clips()
        clip_list = [d for d in clip_list if d.get("status") == Status.DOWNLOADED]
        if self.categories:
            for clip_info in clip_list:
                video_info = self.db.get_video(clip_info.get("video_id"))
                if video_info is None:
                    continue
                clip_info["category"] = video_info.get("category")
            clip_list = [d for d in clip_list if d.get("category") in self.categories]
        print(f"Processing {len(clip_list)} clips", flush=True)
        # clip_paths = [d.get("clip_path") for d in clip_list]
        # if clip_path is not None:  # For debugging
        #     clip_paths = [clip_path]
        shuffle(clip_list)
        for clip in tqdm(clip_list):
            if clip is None:
                continue
            clip_id = clip.get("clip_id")
            category = clip.get("category")
            clip_path = clip.get("clip_path")
            try:
                with Profiler("run_single_clip", num_frames=clip.get("frames"), db=self.db):
                    self.run_single_clip(clip_id, clip_path, category)
            except Exception as e:
                print(f"Fail to process {clip_path}")
                print(f"{type(e).__name__}: {str(e)}", flush=True)
                traceback.print_exc()
                self.reset_clip_status(clip_id)
                continue

    def run_single_clip(self, clip_id, clip_path, category):
        if not self.check_clip_status(clip_path):
            return
        output_dir = Path(self.output_dir) / category / Path(clip_path).stem
        dst_dir = None
        if self.save_local_dir is not None:
            dst_dir = output_dir
            output_dir = Path(self.save_local_dir) / category / Path(clip_path).stem
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = output_dir / "frames"
        num_frames = int(cv2.VideoCapture(clip_path).get(cv2.CAP_PROP_FRAME_COUNT))
        with Profiler("extract_frames", num_frames=num_frames):
            self.extract_frames(clip_path, frames_dir)
        with Profiler("track", num_frames=len(os.listdir(frames_dir))):
            tracking_results, grounding_scores = self.track(frames_dir, category)
            # TODO: modify naive tracker and sam_track output to align with sam2 output, or deprecate both
        if self.save_visualization:
            with Profiler("save_tracking_visualization", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
                self.save_tracking_visualization(
                    tracking_results, frames_dir, output_dir / "tracking_visualization_raw.mp4", grounding_scores
                )
        with Profiler("clean_tracking_results", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.clean_tracking_results(tracking_results)
        with Profiler("filter_overlap", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.filter_overlap(tracking_results)
        with Profiler("filter_small_detection", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.filter_small_detection(tracking_results)
        with Profiler("filter_truncation", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.filter_truncation(tracking_results, border_width=3)
        with Profiler("split_discontinuous_track", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results, grounding_scores = self.split_discontinuous_track(tracking_results, grounding_scores)
        with Profiler("filter_short_track", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.filter_short_track(tracking_results)
        with Profiler("filter_inconsistent_track", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_results = self.filter_inconsistent_track(tracking_results)
        if self.save_visualization and len(self.get_tracked_frame_ids(tracking_results)) > 0:
            with Profiler("save_tracking_visualization", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
                self.save_tracking_visualization(
                    tracking_results, frames_dir, output_dir / "tracking_visualization_processed.mp4", grounding_scores
                )
        with Profiler("get_tracking_crop_boxes", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            tracking_crop_boxes = self.get_tracking_crop_boxes(tracking_results)
        with Profiler("smooth_crop_boxes", num_frames=len(self.get_tracked_frame_ids(tracking_crop_boxes))):
            tracking_crop_boxes = self.smooth_crop_boxes(tracking_crop_boxes)
        with Profiler("fill_missing_masks", num_frames=len(self.get_tracked_frame_ids(tracking_crop_boxes))):
            tracking_results = self.fill_missing_masks(tracking_results, tracking_crop_boxes, frames_dir)
        for frame_id in tracking_crop_boxes.keys():
            assert len(tracking_crop_boxes[frame_id]) == len(tracking_results[frame_id]), \
                f"Frame {frame_id} crop boxes and masks have different length"

        if self.compute_occlusion:
            depths_dir = Path(output_dir) / "depths"
            with Profiler("extract_depths", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
                self.occlusion_processor.extract_depths(tracking_results, clip_path, frames_dir, depths_dir)
        else:
            depths_dir = None
        with Profiler("save_cropped_results", num_frames=len(self.get_tracked_frame_ids(tracking_results))):
            track_ids = self.save_cropped_results(
                tracking_results, tracking_crop_boxes, save_dir=output_dir, frames_dir=frames_dir, depths_dir=depths_dir
            )
        with Profiler(
            "remove_aux_files",
            num_frames=len(os.listdir(frames_dir))+(len(os.listdir(depths_dir) if self.compute_occlusion else 0))
        ):
            shutil.rmtree(frames_dir, ignore_errors=True)
            if self.compute_occlusion:
                shutil.rmtree(depths_dir, ignore_errors=True)
        if len(track_ids) > 0:
            if self.save_local_dir is not None:
                copy_results(output_dir, dst_dir)
                for track_id in track_ids:
                    dst_path = dst_dir / track_id
                    self.db.update_track(track_id, track_path=str(dst_path), location="viscam")
        else:
            shutil.rmtree(output_dir, ignore_errors=True)
        self.update_db_status(clip_id, clip_path)
        return track_ids


if __name__ == "__main__":  # debug
    import warnings
    # warnings.filterwarnings("error")
    from scripts.track_animal import parse_args
    args = parse_args()
    args.version = "2.0.0"
    args.save_local_dir = None
    clip_path = "data/clip_2.0/horse/z525AtRByhQ/z525AtRByhQ_001.mp4"
    output_dir = "data/tmp_tracking_result"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    animal_tracker = AnimalTracker(
        output_dir=output_dir,
        tracking_method=args.tracking_method,
        depth_method=args.depth_method,
        occlusion_batch_size=args.occlusion_batch_size,
        db_path=args.db_path,
        compute_occlusion=args.compute_occlusion,
        tracking_sam_step=args.tracking_sam_step,
        overlap_iou_threshold=args.overlap_iou_threshold,
        truncation_border_width=args.truncation_border_width,
        grounding_text_format=args.grounding_text_format,
        bbox_area_threshold=args.bbox_area_threshold,
        consistency_iou_threshold=args.consistency_iou_threshold,
        sam2_prompt_type=args.sam2_prompt_type,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        crop_margin=args.crop_margin,
        min_scene_len=args.min_scene_len,
        verbose=args.verbose,
        save_visualization=args.save_visualization,
        version=args.version,
        max_track_gap=args.max_track_gap,
        fps=args.fps,
        save_local_dir=args.save_local_dir,
        device=device
    )
    animal_tracker.run(clip_path=clip_path)
