import os
from pathlib import Path
import json
from tqdm import tqdm
import subprocess
from models.youtube_downloader import YouTubeBypassAge




data_dir = Path("data/data_2.0.0.2")
video_id_to_resolution = {}
for metadata_path in tqdm(data_dir.rglob("*metadata.json"), total=sum(1 for _ in data_dir.rglob("*metadata.json"))):
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except:
        print(f"Failed to load {metadata_path}")
        exit()
    if "frame_size_wh" in metadata.keys():
        continue
    video_id = metadata["video_id"]
    if video_id in video_id_to_resolution.keys():
        metadata["frame_size_wh"] = video_id_to_resolution[video_id]
        print(video_id, video_id_to_resolution[video_id])
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    else:
        yt = YouTubeBypassAge(
            "https://youtube.com/watch?v=" + video_id)
        streams = yt.streams.filter(
            type="video", file_extension="mp4", progressive=False,
            custom_filter_functions=[lambda s: not s.video_codec.startswith("av01")]
        ).order_by("resolution")
        stream = streams.last()
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                stream.url
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        video_info = json.loads(result.stdout)
        width = video_info['streams'][0]['width']
        height = video_info['streams'][0]['height']
        video_id_to_resolution[video_id] = [width, height]
        print(video_id, width, height)
        metadata["frame_size_wh"] = [width, height]
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
