import os
import sys

import cv2
import torch
import importlib
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from supervision import mask_to_xyxy
from torchvision.transforms.functional import resize
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor, build_sam2

from externals.Grounded_SAM2.utils.track_utils import sample_points_from_masks
from externals.Grounded_SAM2.utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from models.utils import get_frame_images, Profiler


class GroundedSAM2Tracker:
    def __init__(self, sam2_prompt_type, tracking_sam_step, device):
        checkpoint = "externals/Grounded_SAM2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.tracking_sam_step = tracking_sam_step
        self.sam2_prompt_type = sam2_prompt_type
        self.device = device
        self.mixed_precision = torch.float16

        with Profiler("build SAM2 video predictor"):
            self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        with Profiler("build_sam2"):
            image_model = build_sam2(model_cfg, checkpoint)
        with Profiler("SAM2ImagePredictor"):
            self.image_predictor = SAM2ImagePredictor(image_model)
        with Profiler("AutoProcessor.from_pretrained"):
            self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        # TODO: AutoModelForZeroShotObjectDetection takes too long to load
        with Profiler("AutoModelForZeroShotObjectDetection.from_pretrained"):
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)

    @torch.inference_mode()
    def init_state(
        self, video_path, frames_dir, offload_video_to_cpu=False, offload_state_to_cpu=False,
        img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)
    ):
        """
        Custom function to override self.video_predictor.init_state (SAM2VideoPredictor.init_state)
        Initialize an inference state for SAM2 video predictor
        Support loading from video path or frames directory with different format images
        """
        assert video_path or frames_dir
        image_size = self.video_predictor.image_size
        if frames_dir is not None:
            frame_names = [p for p in os.listdir(frames_dir) if os.path.splitext(p)[-1] == ".png"]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            num_frames = len(frame_names)
            if num_frames == 0:
                raise FileNotFoundError(f"no images found in {frames_dir}")
            img_paths = [os.path.join(frames_dir, frame_name) for frame_name in frame_names]
            img_pil = Image.open(img_paths[0])
            image_height, image_width = img_pil.height, img_pil.width
            frames_tensor = torch.zeros(
                num_frames, 3, image_height, image_width, dtype=torch.float32
            )
            for i, img_path in enumerate(img_paths):
                img_pil = Image.open(img_path)
                img_np = np.array(img_pil.convert("RGB"))
                assert img_np.dtype == np.uint8
                frames_tensor[i] = torch.from_numpy(img_np).permute(2, 0, 1)
        else:  # Loading directly from video mp4
            cap = cv2.VideoCapture(video_path)
            num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if not cap.isOpened():
                print(f"Unable to open video file {video_path}", flush=True)
                return None, None, None
            frames = []
            for _ in num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            frames_np = np.array(frames)
            assert frames_np.dtype == np.uint8
            frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        frames_tensor = frames_tensor / 255.0
        video_height, video_width = frames_tensor.shape[-2:]
        images = resize(frames_tensor, [image_size, image_size])
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        if not offload_video_to_cpu:
            images = images.cuda()
            img_mean = img_mean.cuda()
            img_std = img_std.cuda()
        images -= img_mean
        images /= img_std
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = torch.device(self.device)
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device(self.device)
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        self.video_predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, start_frame_idx=None, max_frame_num_to_track=None, reverse=False,):
        """
        Custom function to override self.video_predictor.propagate_in_video (SAM2VideoPredictor.propagate_in_video)
        Disable tqdm progress bar
        """
        self.video_predictor.propagate_in_video_preflight(inference_state)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self.video_predictor._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.video_predictor.clear_non_cond_mem_around_input and (
            self.video_predicto.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        for frame_idx in processing_order:
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self.video_predictor._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                    current_out, pred_masks = self.video_predictor._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self.video_predictor._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self.video_predictor._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks

    def get_grounding_results(self, video_path=None, frames_dir=None, text_input=None, frame_id=0):
        """
        Get grounding result of the specific frame in video
        """
        image = get_frame_images(video_path=video_path, frames_dir=frames_dir, frame_ids=frame_id)[0]
        inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.6,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
        return results, image

    def __call__first_frame_only(self, video_path=None, frames_dir=None, grounding_text_input=""):
        # TODO: deprecated
        grounding_frame_id = 0
        inference_state = self.init_state(video_path=video_path, frames_dir=frames_dir)
        grounding_results, image = self.get_grounding_results(
            frames_dir=frames_dir, text_input=grounding_text_input, frame_id=grounding_frame_id
        )
        self.image_predictor.set_image(np.array(image))
        input_boxes = grounding_results[0]["boxes"].cpu().numpy()
        objects = grounding_results[0]["labels"]
        if len(objects) == 0:
            video_id = Path(video_path).stem if video_path else Path(frames_dir).parent.stem
            print(f"No detection of '{grounding_text_input}' in {video_id} frame {grounding_frame_id}")
            return {}
        with torch.autocast(device_type=self.device, dtype=self.mixed_precision):  # TODO: check why autocast required
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
        if masks.ndim == 3:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        masks = masks.squeeze(1)

        if self.sam2_prompt_type == "point":
            all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
            for object_id, (label, points) in enumerate(zip(objects, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=grounding_frame_id,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )
        elif self.sam2_prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(objects, input_boxes), start=1):
                with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=grounding_frame_id,
                        obj_id=object_id,
                        box=box,
                    )
        elif self.sam2_prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                    _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=grounding_frame_id,
                        obj_id=object_id,
                        mask=mask
                    )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

        video_segments = {}
        with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        return video_segments

    def __call__(self, video_path=None, frames_dir=None, grounding_text_input=""):
        """
        Track objects in the video using grounded SAM2 algorithm
        Grounding the object in the first frame and prompt the SAM2 model to track the object in the video
        Repeat every self.tracking_sam_step frames and associate the object with the same instance id
        """
        objects_count = 0
        video_segments = {}
        id_to_scores = {}
        sam2_masks = CustomMaskDictionaryModel()
        inference_state = self.init_state(video_path=video_path, frames_dir=frames_dir)
        if os.path.isdir(frames_dir):
            num_frames = len(os.listdir(frames_dir))
            video_id = Path(frames_dir).parent.stem
        else:
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened()
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            video_id = Path(video_path).stem
        for frame_id in range(0, num_frames, self.tracking_sam_step):
            grounding_results, image = self.get_grounding_results(
                frames_dir=frames_dir, text_input=grounding_text_input, frame_id=frame_id
            )
            self.image_predictor.set_image(np.array(image))
            input_boxes = grounding_results[0]["boxes"]  #.cpu().numpy()
            objects = grounding_results[0]["labels"]
            grounding_scores = grounding_results[0]["scores"]
            if len(objects) == 0:
                print(f"No detection of '{grounding_text_input}' in {video_id} frame {frame_id}")
                continue
            with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
            if masks.ndim == 3:
                masks = masks[None]
                scores = scores[None]
                logits = logits[None]
            masks = masks.squeeze(1)

            mask_dict = MaskDictionaryModel(
                promote_type=self.sam2_prompt_type, mask_name=f"{str(frame_id).zfill(8)}_mask.npy"
            )

            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(self.device), box_list=input_boxes.clone().detach(),
                    label_list=objects
                )
            else:
                raise NotImplementedError("SAM 2 video predictor only support mask prompts for new object tracking")

            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count
            )
            if len(mask_dict.labels) == 0:
                print("No object detected in the frame, skip the frame {}".format(frame_id), flush=True)
                continue
            self.video_predictor.reset_state(inference_state)
            with torch.autocast(device_type=self.device, dtype=self.mixed_precision):
                for idx, (object_id, object_info) in enumerate(mask_dict.labels.items()):
                    frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_id,
                        obj_id=object_id,
                        mask=object_info.mask,
                    )
                    if object_id not in id_to_scores:
                        id_to_scores[object_id] = grounding_scores[idx]
                for out_frame_idx, out_obj_ids, out_mask_logits in self.propagate_in_video(
                    inference_state, max_frame_num_to_track=self.tracking_sam_step, start_frame_idx=frame_id
                ):
                    frame_masks = MaskDictionaryModel()
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0)  # .cpu().numpy()
                        # debug: save heatmap
                        # import matplotlib.pyplot as plt
                        # mask_np = out_mask_logits[i].squeeze().cpu().numpy()
                        # plt.figure(figsize=(8, 6))
                        # plt.imshow(mask_np, cmap='coolwarm', aspect='auto')
                        # plt.colorbar()
                        # save_path = f"data/temp_heatmap/{frames_dir.parent.stem}/{str(out_obj_id).zfill(3)}/{str(out_frame_idx).zfill(8)}.png"
                        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # plt.savefig(save_path)
                        # debug end: save heatmap
                        object_info = ObjectInfo(
                            instance_id=out_obj_id, mask=out_mask[0],
                            class_name=mask_dict.get_target_class_name(out_obj_id)
                        )
                        object_info.update_box()
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = str(frame_id).zfill(8)
                        frame_masks.mask_name = f"{image_base_name}_mask.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]
                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = deepcopy(frame_masks)
        tracking_results = {}
        for frame_id, segments in video_segments.items():
            tracking_results[frame_id] = {}
            for object_id, object in segments.labels.items():
                tracking_results[frame_id][object_id] = np.expand_dims(object.mask.cpu().numpy(), axis=0)
        return tracking_results, id_to_scores


class CustomMaskDictionaryModel(MaskDictionaryModel):
    """Custom MaskDictionaryModel class to support removing duplicate detection in update_masks"""
    def update_masks(self, tracking_annotation_dict, iou_threshold=0.8, objects_count=0):
        def contains_bbox(mask1, mask2, soft_margin=5):
            """Check mask1 contains mask2 by comparing bbox with a soft margin"""
            x_min1, y_min1, x_max1, y_max1 = mask_to_xyxy(mask1.unsqueeze(0).cpu().numpy())[0]
            x_min2, y_min2, x_max2, y_max2 = mask_to_xyxy(mask2.unsqueeze(0).cpu().numpy())[0]
            if x_min2 > x_min1 - soft_margin and y_min2 > y_min1 - soft_margin \
            and x_max2 < x_max1 + soft_margin and y_max2 < y_max1 + soft_margin:
                return True
            return False
        def contains_iom2(mask1, mask2, iom2_thresold=0.95):
            """Calculate intersection over mask2 area, checking mask1 contains mask2"""
            intersection = (mask1 * mask2).sum()
            m2 = mask1.sum()
            if (intersection / m2) > iom2_thresold:
                return True
            return False

        updated_masks = {}
        for seg_obj_id, seg_mask in self.labels.items():  # tracking_masks
            flag = 0
            contains_flag = 0
            new_mask_copy = ObjectInfo()
            if seg_mask.mask.sum() == 0:
                continue
            for object_id, object_info in tracking_annotation_dict.labels.items():  # grounded_sam masks
                iou = self.calculate_iou(seg_mask.mask, object_info.mask)  # tensor, numpy
                if iou > iou_threshold:
                    flag = object_info.instance_id
                    new_mask_copy.mask = seg_mask.mask
                    new_mask_copy.instance_id = object_info.instance_id
                    new_mask_copy.class_name = seg_mask.class_name
                    break
                if contains_iom2(object_info.mask, seg_mask.mask):
                    flag = object_info.instance_id
                    contains_flag = 1
                    break
            if not flag:
                objects_count += 1
                flag = objects_count
                new_mask_copy.instance_id = objects_count
                new_mask_copy.mask = seg_mask.mask
                new_mask_copy.class_name = seg_mask.class_name
            if not contains_flag:
                updated_masks[flag] = new_mask_copy
        self.labels = updated_masks
        return objects_count


if __name__ == "__main__":
    video_path = "data/stage1/horse/cr52o96JB60/cr52o96JB60.mp4"
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.category = "horse"
    args.sam2_prompt_type = "mask"
    tracker = GroundedSAM2Tracker(args)
    video_segments = tracker(video_path)

