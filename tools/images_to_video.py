import os
from glob import glob
from tqdm import tqdm
from models.utils import images_to_video, get_all_sequence_dirs


data_dir = "data/data_3.0.0"
src_suffix = "pose.png"
dst_suffix = "pose.mp4"


if __name__ == "__main__":
    all_sequence_dirs = get_all_sequence_dirs(data_dir)
    for sequence_dir in tqdm(all_sequence_dirs):
        data_name = os.path.basename(sequence_dir)
        output_path = os.path.join(sequence_dir, f"{data_name}_{dst_suffix}")
        if os.path.exists(output_path):
            continue
        all_images = sorted(glob(os.path.join(data_dir, f"*{src_suffix}")))
        if len(all_images) == 0:
            continue
        print(f"Processing {data_dir}")
        images_to_video(image_files=all_images, output_path=output_path)
