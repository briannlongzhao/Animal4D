import os
from glob import glob
from tqdm import tqdm
from random import shuffle
from subprocess import run


"""Recursively remove files with given pattern in a directory"""


data_dir = "data/data_3.0.0/"
patterns = [
    "*rgb_overlayed_fauna++.mp4"
    # "*smalify.*"
]


if __name__ == "__main__":  # TODO: refactor this for a more efficient implementation
    if not isinstance(patterns, list):
        patterns = [patterns]
    all_files = []
    for pattern in patterns:
        all_files += list(glob(f"{data_dir}/**/{pattern}", recursive=True))
    shuffle(all_files)
    for file_path in tqdm(all_files):
        if os.path.exists(file_path):
            run(["rm", "-rf", file_path])
            print(f"Removed {file_path}")
