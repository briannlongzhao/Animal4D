import os
from PIL import Image
from tqdm import tqdm
from glob import glob
from random import shuffle


def convert(root_dir, src_suffix, dst_suffix):
    all_files = list(glob(f"{root_dir}/**/*{src_suffix}", recursive=True))
    shuffle(all_files)
    for src_path in tqdm(all_files):
        dst_path = src_path.replace(src_suffix, dst_suffix)
        if os.path.exists(dst_path):
            print(f"Already exists: {dst_path}")
            continue
        with Image.open(src_path) as img:
            if dst_suffix.lower().endswith(".png"):
                dst_img = img.convert('RGBA')
                dst_img.save(dst_path, "PNG")
            elif dst_suffix.lower().endswith(".jpg"):
                dst_img = img.convert('RGB')
                dst_img.save(dst_path, "JPEG", quality=95)
            else:
                raise ValueError(f"Unsupported extension: {os.path.basename(dst_path)}")
        print(f"Converted: {src_path} -> {dst_path}")


if __name__ == "__main__":
    # Update this to the directory you want to process:
    ROOT_DIR = "/viscam/projects/animal_motion/briannlz/video_object_processing/data/magicpony_horse_v2"
    # convert_png_to_jpg(ROOT_DIR)
    convert(ROOT_DIR, src_suffix="rgb.jpg", dst_suffix="rgb.png")
