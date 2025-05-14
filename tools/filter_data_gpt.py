import os
import cv2
from glob import glob
from random import shuffle, choice
from openai import OpenAI
from tqdm import tqdm
from traceback import print_exc
from models.utils import gpt_filter
from database import Database

"Use gpt to filter track data and update database column gpt_filtered"


data_dir = "data/track_3.0.0"
rgb_suffix = "rgb.png"

db_path = "data/database.sqlite"
version = "3.0.0"
db = Database(version=version, db_path=db_path)
use_db = True


if __name__ == "__main__":
    gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if use_db:
        all_tracks = db.get_all_tracks()
        all_data_dir = [track["track_path"] for track in all_tracks]
    else:
        all_image = glob(os.path.join(data_dir, "**", f"*{rgb_suffix}"))
        all_data_dir = list(set([os.path.dirname(image) for image in all_image]))
    shuffle(all_data_dir)
    for track_dir in tqdm(all_data_dir):
        try:
            track_id = os.path.basename(track_dir)
            track_info = db.get_track(track_id)
            if track_info is None:
                continue
            if track_info["gpt_filtered"] is not None:
                continue
            video_id = track_info["video_id"]
            category = db.get_video(video_id)["category"]
            all_images = glob(os.path.join(track_dir, f"*{rgb_suffix}"))
            if len(all_images) == 0:
                print(f"No images found in {track_dir}")
                continue
            image = cv2.imread(choice(all_images))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            keep = gpt_filter(gpt_client=gpt_client, image=image, category=category)
            if keep:
                db.update_track(track_id, gpt_filtered=True)
            else:
                db.update_track(track_id, gpt_filtered=False)
        except:
            print_exc()
            continue
