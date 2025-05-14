import os
from tqdm import tqdm
from glob import iglob

"""Rename files in a directory by replacing source suffix to destination suffix"""

data_dir = "data/track_3.0.0"
src_suffix = "pose.txt"
dst_suffix = "keypoint.txt"

if __name__ == "__main__":
    for file_path in tqdm(iglob(f"{data_dir}/**/*{src_suffix}", recursive=True)):
        dst_path = str(file_path).replace(src_suffix, dst_suffix)
        os.rename(file_path, dst_path)
        print(f"Renamed {file_path} to {dst_path}")
