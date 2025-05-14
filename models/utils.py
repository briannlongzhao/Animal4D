import os
import cv2
import sys
import torch
import numpy as np
from subprocess import run
from glob import glob
from time import time
from random import shuffle
from PIL import Image
from pathlib import Path
from datetime import datetime
from base64 import b64encode
from io import BytesIO
from time import sleep
from tqdm import tqdm


class Logger(object):
    """
    Custom logger that prints to both stdout and a log file
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        os.makedirs(Path(file_path.parent), exist_ok=True)
        self.log_file = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


class Profiler:  # TODO: remove db
    """Profiler context manager, optionally give num_frames or list of video/clip/track to retrieve num_frames"""
    def __init__(
        self, name, db=None, num_frames=None,
        video_ids=None, video_paths=None, clip_ids=None, clip_paths=None, track_ids=None, track_paths=None,
    ):
        self.name = name
        self.num_frames = None
        if num_frames is not None:
            self.num_frames = num_frames
        elif video_ids or video_paths or clip_ids or clip_paths or track_ids or track_paths:
            assert db is not None, "db must be provided to retrieve num_frames"
            if video_ids or video_paths:
                if video_paths:
                    video_ids = [Path(video_path).stem for video_path in video_paths]
                self.num_frames = sum([db.get_video(video_id)["frames"] for video_id in video_ids])
            if clip_ids or clip_paths:
                if clip_paths:
                    clip_ids = [Path(clip_path).stem for clip_path in clip_paths]
                self.num_frames = sum([db.get_clip(clip_id)["frames"] for clip_id in clip_ids])
            if track_ids or track_paths:
                if track_paths:
                    track_ids = [Path(track_path).stem for track_path in track_paths]
                self.num_frames = sum([db.get_track(track_id)["length"] for track_id in track_ids])

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapse = time() - self.start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.name}: {elapse:.2f}s", end='')
        if self.num_frames is not None:
            print(f" ({self.num_frames/elapse:.2f} frames/s)", end='')
        print()


def get_frame_images(video_path=None, frames_dir=None, frame_ids=0):
    """
    Get frame images from a video
    Return list of RGB PIL image in (w h c)
    """
    if isinstance(frame_ids, int):
        frame_ids = [frame_ids]
    assert video_path or frames_dir
    images = []
    if frames_dir is not None:
        image_paths = [Path(frames_dir) / f"{str(frame_id).zfill(8)}.png" for frame_id in frame_ids]
        for image_path in image_paths:
            assert image_path.is_file(), image_path
            images.append(Image.open(image_path))
    else:
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame))
        cap.release()
    return images


def gpu_memory_usage():
    """Get GPU memory usage proportion in MB"""
    reserved_memory = torch.cuda.memory_reserved()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    return reserved_memory / total_memory


def timecode_to_second(timecode):
    h, m, s_ms = str(timecode).split(':')
    s, ms = s_ms.split('.')
    seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    return seconds


def crop(image, crop_box, resize_width, resize_height):
    """Crop and pad array (h,w,...) based on crop box (xyxy) in original array then resize"""
    if image.ndim == 3:
        c = image.shape[2]
    elif image.ndim == 2:
        c = 1
    x_min, y_min, x_max, y_max = crop_box
    crop_height, crop_width = y_max - y_min, x_max - x_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    left, top = int(x_center - crop_width / 2), int(y_center - crop_height / 2)
    right, bottom = int(x_center + crop_width / 2), int(y_center + crop_height / 2)
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - image.shape[1]), max(0, bottom - image.shape[0])
    left, top = max(0, left), max(0, top)
    right, bottom = min(image.shape[1], right), min(image.shape[0], bottom)
    cropped_image = image[top:bottom, left:right]
    padded_image = np.zeros((crop_height, crop_width, c), dtype=image.dtype).squeeze()
    padded_image[pad_top:pad_top + cropped_image.shape[0], pad_left:pad_left + cropped_image.shape[1]] = cropped_image
    padded_image = cv2.resize(padded_image, (resize_width, resize_height))
    return padded_image


def gpt_filter(gpt_client, category, image, max_retry=3, model="gpt-4o-mini"):
    def np_to_b64(image_array):
        image_array = Image.fromarray(image_array)
        image_array = image_array.resize((256, 256))
        buffer = BytesIO()
        image_array.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_str = b64encode(buffer.getvalue()).decode("utf-8")
        return base64_str
    prompt = f"Does this image show a realistic photo of a {category} without any occlusion? Answer yes or no only."
    base64_image = np_to_b64(image)
    retry = 0
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
    ]}]
    while True:
        if retry > max_retry:
            print("GPT filter failed", flush=True)
            return False
        try:
            response = gpt_client.chat.completions.create(
                messages=messages, model=model
            ).choices[0].message.content
            if "yes" in response.lower():
                print("GPT filter passed", flush=True)
                return True
            else:
                print("GPT filter failed", flush=True)
                return False
        except Exception as e:
            print(f"{type(e).__name__}: {str(e)}", flush=True)
            sleep(2)
            retry += 1
            continue


def images_to_video(image_files, output_path, fps=10):
    frame = cv2.imread(str(image_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    print(f"Saving video to {output_path}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        out.write(frame)
    out.release()


def validate_tensor_to_device(x, device=None):
    if type(x) is not torch.Tensor:
        return x
    elif torch.any(torch.isnan(x)):
        return None
    elif device is None:
        return x
    else:
        return x.to(device)


def validate_all_to_device(batch, device=None):
    return tuple(validate_tensor_to_device(x, device) for x in batch)


def get_all_sequence_dirs(data_dir, image_suffix="rgb.png", sorted=False):
    result = set()
    for root, dirs, files in tqdm(os.walk(data_dir, followlinks=True), desc="Getting all sequence dirs"):
        for f in files:
            if f.endswith(image_suffix):
                result.add(root)
    result = list(result)
    if sorted:
        result.sort()
    else:
        shuffle(result)
    return result


def copy_results(src_dir, dst_dir, suffixes):
    """Copy all files from src_dir to dst_dir that match any of the given suffixes"""
    cmd = ["rsync", "-avz", "--include=*/"]
    for s in suffixes:
        pattern = f"--include=*{s}"
        cmd.append(pattern)
    cmd.append("--exclude=*")
    cmd.append(src_dir.rstrip('/') + '/')
    cmd.append(dst_dir)
    run(cmd, check=True)


def mask_diff(mask1, mask2):
    """Return difference of two masks, where pixels are 1 if different, 0 if same"""
    assert mask1.shape == mask2.shape
    if isinstance(mask1, torch.Tensor):
        return (mask1 - mask2).abs().clamp(0, 1)
    elif isinstance(mask1, np.ndarray):
        return np.abs(mask1 - mask2).clip(0, 1)
    else:
        raise NotImplementedError


def remove_background(image, bg_value=0, c_dim=-3):
    """Add alpha channel to image where background pixels are transparent"""
    if isinstance(image, torch.Tensor):
        if image.shape[c_dim] == 1:
            image = torch.cat([image, image, image], dim=c_dim)
        bg = torch.all(image == bg_value, dim=c_dim, keepdim=True)
        alpha = (~bg).float()
        image = torch.cat([image, alpha], dim=c_dim)
    elif isinstance(image, np.ndarray):
        raise NotImplementedError
    else:
        raise NotImplementedError
    return image


def draw_keypoints(image, keypoint, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    # Tensor image input in (C,H,W), normalized in [0,1], keypoit in (N,3), normalized in [0,1]
    assert isinstance(image, torch.Tensor) and len(image.shape) == 3 and image.shape[0] == 3
    # assert len(keypoint.shape) == 2 and keypoint.max() <= 1.0 and keypoint.min() >= 0.0
    h, w = image.shape[-2:]
    image_np = (image.clone() * 255).permute(1, 2, 0).clamp(0, 255).contiguous().cpu().numpy().astype(np.uint8)
    for i, (x, y) in enumerate(keypoint[:, :2]):
        x, y = int(x.item() * w), int(y.item() * h)
        cv2.circle(image_np, center=(x, y), radius=radius, color=circle_color, thickness=-1)
        cv2.putText(
            image_np, text=str(i), org=(x + radius + 1, y - radius - 1),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5, color=text_color, thickness=1, lineType=cv2.LINE_AA
        )
    image_out = torch.from_numpy(image_np).permute(2, 0, 1) / 255
    return image_out


SMAL_JOINTS = {
    10: "LF feet", 9: "LF middle", 8: "LF top",
    20: "LR feet", 19: "LR middle", 18: "LR top",
    14: "RF feet", 13: "RF middle", 12: "RF top",
    24: "RR feet", 23: "RR middle", 22: "RR top",
    25: "tail start", 31: "tail end",
    33: "left ear base", 34: "right ear base",
    35: "nose", 36: "chin",
    38: "left ear tip", 37: "right ear tip",
    39: "left eye", 40: "right eye",
    15: "withers", 28: "tail middle"
}

VITPOSE_JOINTS = {
    0: "left eye", 1: "right eye", 2: "nose",
    3: "withers", 4: "tail start",
    5: "LF top", 6: "LF middle", 7: "LF feet",
    8: "RF top", 9: "RF middle", 10: "RF feet",
    11: "LR top", 12: "LR middle", 13: "LR feet",
    14: "RR top", 15: "RR middle", 16: "RR feet",
}

FAUNA_JOINTS = {
    0: "nose", 1: "neck start", 2: "neck end", 3: "withers",
    4: "middle", 5: "hip", 6: "tail start", 7: "tail start/middle", 8: "tail middle/end",
    9: "LF top", 10: "LF middle", 11: "LF feet",
    12: "RF top", 13: "RF middle", 14: "RF feet",
    15: "LR top", 16: "LR middle", 17: "LR feet",
    18: "RR top", 19: "RR middle", 20: "RR feet",
}
