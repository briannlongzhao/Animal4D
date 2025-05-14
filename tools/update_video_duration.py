from database import Database
import cv2
from tqdm import tqdm

version = "3.0.0"

db = Database(db_path="data/database.sqlite", version=version)
all_videos = db.get_all_videos()
for video in tqdm(all_videos):
    video_path = video.get("video_path")
    video_id = video.get("video_id")
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps
    cap.release()
    db.update_video(video_id, duration=duration, fps=fps, frames=frames)



