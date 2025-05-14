import os
import shutil
from tqdm import tqdm
from pathlib import Path
from rclone_python import rclone

version = "0.0.0"
src_dir = Path(f"data/track_{version}/horse/")
dst_dir = Path(f"data/track_{version}_vis/")
remote_name = "google-drive-stanford"

"""Copy all processed visualizations into one single directory"""
assert os.path.exists(src_dir), f"Source directory {src_dir} does not exist"
print("Clearing destination directory")
shutil.rmtree(dst_dir, ignore_errors=True)

video_paths = []
dst_paths = []
os.makedirs(dst_dir, exist_ok=True)
for video_dir in tqdm(src_dir.iterdir(), total=len(os.listdir(src_dir)), desc="scanning directories"):
    if not video_dir.is_dir():
        continue
    os.makedirs(dst_dir/video_dir.name, exist_ok=True)
    for object_dir in video_dir.iterdir():
        if object_dir.suffix == ".mp4":
            shutil.copy(object_dir, dst_dir/video_dir.name)
        if not object_dir.is_dir():
            continue
        for file in object_dir.iterdir():
            if file.suffix == ".mp4":
                video_paths.append(file)
                dst_paths.append(dst_dir/video_dir.name/f"{file.stem}.mp4")

for video_path, dst_path in tqdm(zip(video_paths, dst_paths), desc="Copying visualizations"):
    shutil.copy(video_path, dst_path)

"""Upload visualizations to google drive"""
assert(rclone.is_installed()), "rclone is not installed"
assert rclone.check_remote_existing(remote_name), f"Remote {remote_name} does not exist"
rclone.sync(str(dst_dir), f"{remote_name}:{dst_dir.name}", show_progress=True)
