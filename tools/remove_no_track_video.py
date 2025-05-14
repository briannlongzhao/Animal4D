import os
from tqdm import tqdm
from random import shuffle
from database import Database, Status

"""
Remove all videos with no track processed in database
"""


db_path = "data/database.sqlite"
version = "3.0.0"


if __name__ == "__main__":
    db = Database(version=version, db_path=db_path)
    unprocessed_clips = db.get_all_clips(condition=f"status = '{Status.DOWNLOADED}'")
    assert len(unprocessed_clips) == 0, "All clips should be processed before running this script"
    all_videos = db.get_all_videos(condition=f"status = '{Status.PROCESSED}'")
    shuffle(all_videos)
    discarded = 0
    for video in tqdm(all_videos):
        if video.get("status") != "processed":
            continue
        video_id = video.get("video_id")
        all_tracks = db.get_all_tracks(condition=f"video_id = '{video_id}'")
        if len(all_tracks) == 0:
            print(f"Discard {video_id}")
            discarded += 1
            print(discarded, flush=True)
            db.remove_track(video_id=video_id)
            db.remove_clip(video_id=video_id)
            video_path = video.get("video_path")
            if video_path is not None and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass
            db.discard_video(video_id)
            db.update_video(video_id, video_path=None)
