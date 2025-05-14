from database import Database, Status
from tqdm import tqdm

"""
Reset clip status for all clips that have no detected tracks
Caused by no detection due to incompatible GPUs
Will reset all clips to DOWNLOADED status
RUN WITH CAUTION since it will also reset large amount of clips that indeed have no tracks saved after filtering
"""

clip_ids = [
]
version = "3.0.0"
db_path = "data/database.sqlite"


if __name__ == "__main__":
    db = Database(version=version, db_path=db_path)
    if not clip_ids:
        all_clips = db.get_all_clips()
        all_clip_ids = [clip.get("clip_id") for clip in all_clips if clip.get("status") == Status.PROCESSED]
        all_reset_clip_ids = []
        for clip_id in tqdm(all_clip_ids, desc="Checking no result clips"):
            all_tracks = db.get_all_tracks(condition=f"clip_id = '{clip_id}'")
            if len(all_tracks) == 0:
                all_reset_clip_ids.append(clip_id)
    else:
        all_reset_clip_ids = clip_ids
    print(f"Resetting {len(all_reset_clip_ids)} clips")
    for clip_id in tqdm(all_reset_clip_ids):
        db.update_clip(clip_id, status=Status.DOWNLOADED)
