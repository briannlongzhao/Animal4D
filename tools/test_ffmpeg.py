import os
import cv2
import traceback
import numpy as np
from pathlib import Path
from shutil import rmtree
from scenedetect import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode

"""
Test capability of FFmpeg to write video in avc1 codec using opencv
"""
test_dir = "test_ffmpeg"
os.makedirs(test_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'avc1')

try:
    out = cv2.VideoWriter(os.path.join(test_dir, "temp.mp4"), fourcc, 20.0, (640, 480))
    for i in range(100):
        frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
except Exception as e:
    print(f"{type(e).__name__}: {str(e)}", flush=True)
    traceback.print_exc()

"""
Test capability of FFmpeg to read and split video using pyscenedetect
"""
scene_list = [
    (FrameTimecode(0, fps=20), FrameTimecode(50, fps=20)),
    (FrameTimecode(50, fps=20), FrameTimecode(100, fps=20))
]
try:
    retval = split_video_ffmpeg(
        os.path.join(test_dir, "temp.mp4"), scene_list=scene_list, output_dir=Path(test_dir), show_progress=True,
        output_file_template="${VIDEO_NAME}_${SCENE_NUMBER}.mp4"
    )
    assert retval == 0, f"FFmpeg returned {retval}"
except Exception as e:
    print(f"{type(e).__name__}: {str(e)}", flush=True)
    traceback.print_exc()

rmtree(test_dir)
print("Done")
