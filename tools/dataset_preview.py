import os
import random
import math
from pathlib import Path
from glob import glob
from moviepy.editor import (
    VideoFileClip,
    clips_array,
    concatenate_videoclips,
    ColorClip,
    CompositeVideoClip,
    ImageClip
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np

"""Randomly sample a few videos from a dataset directory and create a looping video grid"""

# Configuration
data_dir = "data/data_3.0.0"
output_video = f"data/{Path(data_dir).name}_preview.mp4"
num_rows = 10
num_cols = 15
grid_duration = 10
margin = 5
video_suffix = "rgb_masked.mp4"



def loop_video(clip, duration):
    """
    Loop the video clip to match the desired duration.

    Parameters:
    - clip (VideoFileClip): The video clip to loop.
    - duration (float): Desired duration in seconds.

    Returns:
    - VideoFileClip: The looped video clip.
    """
    if clip.duration >= duration:
        return clip.subclip(0, duration)
    else:
        n_loops = math.ceil(duration / clip.duration)
        return concatenate_videoclips([clip] * n_loops).subclip(0, duration)


def create_blank_clip(width, height, duration, color=(0, 0, 0)):
    """
    Create a blank (solid color) video clip.

    Parameters:
    - width (int): Width of the blank clip.
    - height (int): Height of the blank clip.
    - duration (float): Duration of the blank clip in seconds.
    - color (tuple): RGB color tuple for the blank clip.

    Returns:
    - ColorClip: The blank video clip.
    """
    return ColorClip(size=(width, height), color=color).set_duration(duration)


def add_text_overlay(clip, text):
    """
    Add a text overlay to the upper left corner of the video clip using Pillow.

    Parameters:
    - clip (VideoFileClip): The video clip to add text to.
    - text (str): The text to overlay.

    Returns:
    - CompositeVideoClip: The video clip with text overlay.
    """
    # Create a transparent image for the text
    txt_image = Image.new('RGBA', (clip.w, clip.h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_image)

    # Define text properties
    font_size = 30  # Adjust as needed
    try:
        # Attempt to use a common TrueType font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # If the font file is not found, use the default font
        font = ImageFont.load_default()

    # Calculate text size using textbbox for better accuracy
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Define position: upper left corner with 10px padding
    position = (10, 10)

    # Add text to the image
    draw.text(position, text, font=font, fill=(255, 255, 255, 255))  # White color

    # Convert the PIL image to a MoviePy ImageClip by converting to a NumPy array
    txt_clip = ImageClip(np.array(txt_image)).set_duration(clip.duration)

    # Composite the text clip over the original video clip
    video_with_text = CompositeVideoClip([clip, txt_clip])

    return video_with_text


def resize_clip(clip, target_size):
    """
    Resize the clip to the target size while maintaining aspect ratio.

    Parameters:
    - clip (VideoFileClip): The video clip to resize.
    - target_size (int): The target width and height for the grid cell (square).

    Returns:
    - VideoFileClip: The resized video clip.
    """
    # Since the target is square and assuming the original video is square,
    # simply resize to (target_size, target_size)
    return clip.resize((target_size, target_size))


def create_video_grid(
        video_paths, num_rows, num_cols, output_path, grid_duration=30, margin=5
):
    """
    Create a video grid from a list of video paths with preserved aspect ratios and text overlays.

    Parameters:
    - video_paths: List of paths to video files.
    - num_rows: Number of rows in the grid.
    - num_cols: Number of columns in the grid.
    - output_path: Path to save the final grid video.
    - grid_duration: Duration of the final video in seconds.
    - margin: Margin between videos in pixels.
    """
    total_cells = num_rows * num_cols
    available_videos = len(video_paths)

    # If more videos than grid cells, randomly select the required number
    if available_videos > total_cells:
        sampled_videos = random.sample(video_paths, total_cells)
    else:
        sampled_videos = video_paths.copy()

    # If fewer videos than grid cells, fill the rest with None (to be replaced with blank clips)
    while len(sampled_videos) < total_cells:
        sampled_videos.append(None)  # Placeholder for blank clips

    # Define individual grid cell size (square)
    desired_clip_size = 320  # Example size; adjust as needed

    # Calculate final grid resolution based on rows and cols
    final_width = num_cols * desired_clip_size + (num_cols - 1) * margin
    final_height = num_rows * desired_clip_size + (num_rows - 1) * margin

    # Load and process all clips
    processed_clips = []
    for idx, path in enumerate(sampled_videos):
        if path is not None:
            try:
                clip = VideoFileClip(path)
                print(f"Processing video {idx + 1}/{total_cells}: {path}")
                # Loop the clip to match grid_duration
                clip = loop_video(clip, grid_duration)
                # Resize to maintain aspect ratio without padding
                clip = resize_clip(clip, desired_clip_size)
                # Mute audio to prevent overlapping sounds
                clip = clip.without_audio()
                # Add text overlay (using the video filename as identifier)
                track_id = Path(path).stem
                clip = add_text_overlay(clip, track_id)
                processed_clips.append(clip)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Append a blank clip in case of error
                blank_clip = create_blank_clip(desired_clip_size, desired_clip_size, grid_duration)
                # Add "Error" text to blank clip
                blank_clip = add_text_overlay(blank_clip, "Error")
                processed_clips.append(blank_clip)
        else:
            # Create a blank clip
            print(f"Adding blank clip at position {idx + 1}")
            blank_clip = create_blank_clip(desired_clip_size, desired_clip_size, grid_duration)
            # Add "No Video" text to blank clip
            blank_clip = add_text_overlay(blank_clip, "No Video")
            processed_clips.append(blank_clip)

    # Arrange clips into rows and columns
    grid = []
    for row in range(num_rows):
        row_clips = []
        for col in range(num_cols):
            idx = row * num_cols + col
            row_clips.append(processed_clips[idx])
        grid.append(row_clips)

    # Create the final grid using clips_array without unsupported parameters
    final_clip = clips_array(grid).set_duration(grid_duration)

    # Write the final video
    print(f"Writing final video to {output_path}")
    final_clip.write_videofile(output_path, fps=24, codec='libx264')

    # Close all clips to release resources
    for clip in processed_clips:
        if clip is not None:
            clip.close()
    final_clip.close()


if __name__ == "__main__":
    video_files = glob(os.path.join(data_dir, "**", f"*{video_suffix}"), recursive=True)
    video_files = random.sample(video_files, num_rows * num_cols)
    if not video_files:
        print(f"No video files found in {data_dir} with suffix {video_suffix}.")
        exit(1)
    create_video_grid(
        video_paths=video_files,
        num_rows=num_rows,
        num_cols=num_cols,
        output_path=output_video,
        grid_duration=grid_duration,
        margin=margin
    )
    print(f"Video grid saved to {output_video}")
