import os.path
from pathlib import Path
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from models.utils import get_all_sequence_dirs
from database import Database

data_dir = Path("data/track_3.0.0")  # Should have all categories as subdirectories
image_suffix = "rgb.png"
use_database = True  # Use database only if stating unfiltered tracks
version = "3.0.0"
db_path = f"data/database.sqlite"

if use_database:
    assert "track" in str(data_dir)


def pie_chart(data, save_path="data/temp_pie.pdf"):
    labels = list(data.keys())
    sizes = list(data.values())
    # explode = [0.03] * len(labels)  # 0.03 means 3% offset for every slice
    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        # explode=explode,
        shadow=False,
        textprops={'fontsize': 16}
    )
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    print("Saved pie chart to", save_path)


def bar_chart(data, save_path="data/temp_bar.pdf"):
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=False)
    categories, values = zip(*sorted_items)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({"Category": categories, "Values": values})
    sns.barplot(data=df, y="Category", x="Values", order=categories)
    ax = plt.gca()
    for container in ax.containers:
        for bar in container.patches:
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            label = f"{int(x)}"
            ax.text(x + 0.5, y, label, va='center', fontsize=10)
    ax.set_xlabel("# frames", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    print("Saved bar chart to", save_path)


if __name__ == "__main__":
    category_to_frames = {}
    category_to_tracks = {}
    if use_database:  # For track directory, not for data directory
        db = Database(db_path=db_path, version=version)
        all_tracks = db.get_all_tracks()
        for track in tqdm(all_tracks):
            track_path = track["track_path"]
            if not os.path.exists(track_path):
                print(f"Track {track_path} does not exist", flush=True)
                continue
            if track["gpt_filtered"] == False:
                print(f"Track {track_path} is not gpt_filtered", flush=True)
                continue
            video_id = track["video_id"]
            # TODO: better way to get category
            category = track_path.split('/')[-3]
            if category not in category_to_frames:
                category_to_frames[category] = 0
                category_to_tracks[category] = 0
            category_to_tracks[category] += 1
            category_to_frames[category] += track["length"]
    else:
        for cat_dir in tqdm(list(data_dir.iterdir())):
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            if category not in category_to_frames:
                category_to_frames[category] = 0
                category_to_tracks[category] = 0
            all_sequence_dirs = get_all_sequence_dirs(cat_dir)
            category_to_tracks[category] = len(all_sequence_dirs)
            for sequence_dir in all_sequence_dirs:
                num_frames = len(list(glob(f"{sequence_dir}/*{image_suffix}", recursive=True)))
                category_to_frames[category] += num_frames

    for category in category_to_frames.keys():
        print(f"{category}:\t{category_to_tracks[category]} tracks,\t{category_to_frames[category]} frames")
    print(f"total:\t{sum(category_to_tracks.values())} tracks,\t{sum(category_to_frames.values())} frames")

    bar_chart(category_to_frames)
    pie_chart(category_to_frames)




