import os
from tqdm import tqdm
from random import shuffle
from subprocess import run

from database import Database
from tools.data_ids import filter_video_ids, filter_clip_ids, filter_track_ids


"""
Remove video/clip/track from database and disk, using manually input ids in filter_ids.py
"""

version = "3.0.0"
db_path = "data/database.sqlite"


if __name__ == "__main__":
    db = Database(version=version, db_path=db_path)
    shuffle(filter_video_ids)
    shuffle(filter_clip_ids)
    shuffle(filter_track_ids)

    # Remove track from database and disk
    for track_id in tqdm(filter_track_ids, desc="Removing tracks"):
        db.remove_track(track_id=track_id)

    # Remove clip and track from database and disk
    for clip_id in tqdm(filter_clip_ids, desc="Removing clips"):
        db.remove_track(clip_id=clip_id)
        db.remove_clip(clip_id=clip_id)

    # Remove video from disk
    # Discard video in database, remove clip and track from database and disk
    for video_id in tqdm(filter_video_ids, desc="Removing videos"):
        video = db.get_video(video_id)
        if video is not None:
            video_path = video.get("video_path")
            if video_path is not None and os.path.exists(video_path):
                try:
                    run(["rm", "-rf", video_path])
                except:
                    pass
        # Discard video in database, remove clip and track from database and disk
        db.remove_track(video_id=video_id)
        print(f"Removing tracks and clips of video {video_id}")
        db.remove_clip(video_id=video_id)
        print(f"Removing clips of video {video_id}")
        db.discard_video(video_id)
        print(f"Discarding video {video_id}")
        db.update_video(video_id, video_path=None)
        print(f"Updating video path {video_id}")
