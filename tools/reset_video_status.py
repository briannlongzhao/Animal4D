from database import Database, Status
from tqdm import tqdm
from shutil import rmtree

"""
Reset status for given videos
Maybe caused by collisions in preprocessing stage, and clips cannot be found in tracking stage
All videos will be in DOWNLOADED status after reset
TODO: check why preprocessing may have collisions
"""

video_ids = [
]
version = "3.0.0"
db_path = "data/database.sqlite"


if __name__ == "__main__":
    db = Database(version=version, db_path=db_path)
    for video_id in tqdm(video_ids):
        video_info = db.get_video(video_id)
        if video_info is None:
            print(f"Video {video_id} not found in database")
            continue
        # Remove all tracks on disk
        all_tracks = db.get_all_tracks(condition=f"video_id = '{video_id}'")
        for track in all_tracks:
            track_path = track.get("track_path")
            print(f"Removing track {track_path}")
            rmtree(track_path, ignore_errors=True)
        # Remove all clips on disk
        all_clips = db.get_all_clips(condition=f"video_id = '{video_id}'")
        for clip in all_clips:
            clip_path = clip.get("clip_path")
            if clip_path is not None:
                print(f"Removing clip {clip_path}")
                rmtree(clip_path, ignore_errors=True)
        # Reset database status
        db.remove_track(video_id=video_id)
        db.remove_clip(video_id=video_id)
        db.update_video(video_id, status=Status.DOWNLOADED)
