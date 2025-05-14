import os
import cv2
import sqlite3
from tqdm import tqdm
from enum import Enum
from time import sleep
from pathlib import Path
from functools import wraps
from datetime import datetime
from socket import gethostname
from configargparse import ArgumentParser
from subprocess import run


class Status(str, Enum):
    """Enumeration of video and clip status in database"""
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    INACCESSIBLE = "inaccessible"
    DISCARDED = "discarded"


def config_db_path():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, default="configs/config.yml")
    parser.add_argument("--db_path", type=str)
    args, _ = parser.parse_known_args()
    return args.db_path


def convert_to_dicts(cursor, rows):
    """Convert query result to list of dictionaries with column names as keys"""
    column_names = [description[0] for description in cursor.description]
    result = []
    for row in rows:
        row_dict = dict(zip(column_names, row))
        result.append(row_dict)
    return result


def parse_version(version):
    """Parse version string to tuple"""
    # TODO: maybe add more parts
    if not version:
        return {}
    version_list = version.split('.')
    assert len(version_list) >= 3, "Version string must have 3 parts"
    version_dict = {}
    if len(version_list) >= 1:
        version_dict["video"] = version_list[0]
    if len(version_list) >= 2:
        version_dict["clip"] = f"{version_list[0]}.{version_list[1]}"
    if len(version_list) >= 3:
        version_dict["track"] = f"{version_list[0]}.{version_list[1]}.{version_list[2]}"
    if len(version_list) >= 4:
        version_dict["data"] = f"{version_list[0]}.{version_list[1]}.{version_list[2]}.{version_list[3]}"
    return version_dict


def make_video_table(version=None, db_path=config_db_path()):
    """Initialize video table"""
    video_version = parse_version(version).get("video")
    table_name = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            video_id VARCHAR (50) NOT NULL UNIQUE,
            category VARCHAR (20) NOT NULL,
            status VARCHAR (20) NOT NULL,
            video_path VARCHAR (255) UNIQUE,
            duration REAL,
            fps REAL,
            frames INTEGER,
            title TEXT,
            keywords TEXT,
            query_text TEXT NOT NULL,
            access_date_time TEXT NOT NULL,
            inaccessible_reason TEXT,
            PRIMARY KEY (video_id)
        );
    """)
    db.commit()
    db.close()


def make_clip_table(version=None, db_path=config_db_path()):
    """Initialize clip table"""
    clip_version = parse_version(version).get("clip")
    table_name = f"clip_{clip_version}" if clip_version else "clip"
    db = sqlite3.connect(db_path)  # TODO: decide start_time or start_frame
    db.cursor().execute(f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            clip_id VARCHAR (50) NOT NULL UNIQUE,
            video_id VARCHAR (20) NOT NULL,
            status VARCHAR (20) NOT NULL,
            clip_path VARCHAR (255) UNIQUE,
            start_time REAL NOT NULL,
            start_frame INTEGER NOT NULL,
            duration REAL NOT NULL,
            frames INTEGER NOT NULL,
            process_date_time TEXT,
            filter_reason TEXT,
            PRIMARY KEY (clip_id),
            FOREIGN KEY (video_id) REFERENCES video (video_id)
        );
    """)
    db.commit()
    db.close()


def make_track_table(version=None, db_path=config_db_path()):
    """Initialize track table"""
    track_version = parse_version(version).get("track")
    table_name = f"track_{track_version}" if track_version else "track"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            track_id VARCHAR (50) NOT NULL UNIQUE,
            clip_id VARCHAR (50) NOT NULL,
            video_id VARCHAR (20) NOT NULL,
            track_path VARCHAR (255) UNIQUE,
            status VARCHAR (20) NOT NULL,
            length INTEGER NOT NULL,
            occlusion REAL,
            flow REAL,
            process_date_time TEXT,
            filter_reason TEXT,
            location TEXT,
            PRIMARY KEY (track_id),
            FOREIGN KEY (clip_id) REFERENCES clip (clip_id),
            FOREIGN KEY (video_id) REFERENCES video (video_id)
        );
    """)
    db.commit()
    db.close()


def make_data_table(version=None, db_path=config_db_path()):
    """Initialize data table"""
    data_version = parse_version(version).get("data")
    table_name = f"data_{data_version}" if data_version else "data"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        CREATE TABLE IF NOT EXISTS '{table_name}' (
            data_id VARCHAR (50) NOT NULL UNIQUE,
            feature_status VARCHAR (20),
            data_path VARCHAR (255) NOT NULL UNIQUE,
            split VARCHAR (20) NOT NULL,
            length INTEGER NOT NULL,
            occlusion REAL,
            flow REAL,
            process_date_time TEXT,
            FOREIGN KEY (data_id) REFERENCES track (track_id)
        );
    """)
    db.commit()
    db.close()


def insert_video(
    video_id, category, query_text, duration=None, fps=None, frames=None, title=None, keywords=None,
    video_path=None, reason=None, version=None, db_path=config_db_path()
):
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    if reason is None:
        db.cursor().execute(
            f"""
                INSERT OR REPLACE INTO '{video_table}' (
                    video_id, status, category, video_path, duration, fps, frames, 
                    title, keywords, query_text, access_date_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                video_id, Status.DOWNLOADED, category, str(video_path), duration, fps, int(frames), title, keywords,
                query_text, datetime.now()
            )
        )
    else:
        db.cursor().execute(
            f"""
                INSERT OR REPLACE INTO '{video_table}' (
                    video_id, status, category, duration, fps, frames, title, keywords, query_text, access_date_time, 
                    inaccessible_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                video_id, Status.INACCESSIBLE, category, duration, fps, int(frames), title, keywords, query_text,
                datetime.now(), reason
            )
        )
    db.commit()
    db.close()


def insert_clip(  # TODO: decide start_time or start_frame
    clip_id, video_id, clip_path, start_time, start_frame, duration, frames, filter_reason=None,
    version=None, db_path=config_db_path()
):
    clip_version = parse_version(version).get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    db = sqlite3.connect(db_path)
    if filter_reason is None:
        db.cursor().execute(
            f"""
                INSERT OR REPLACE INTO '{clip_table}' 
                (clip_id, video_id, status, clip_path, start_time, start_frame, duration, frames, process_date_time) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                clip_id, video_id, Status.DOWNLOADED, str(clip_path),
                str(start_time), int(start_frame), str(duration), int(frames), datetime.now()
            )
        )
    else:
        db.cursor().execute(
            f"""
                INSERT OR REPLACE INTO '{clip_table}' 
                (
                    clip_id, video_id, status, clip_path, start_time, start_frame, duration, frames, 
                    process_date_time, filter_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                clip_id, video_id, Status.DISCARDED, str(clip_path),
                str(start_time), int(start_frame), duration, int(frames), datetime.now(), filter_reason
            )
        )
    db.commit()
    db.close()


def insert_track(
    track_id, clip_id, video_id, track_path, length, occlusion=None, flow=None, filter_reason=None,
    version=None, db_path=config_db_path()
):
    track_version = parse_version(version).get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    location = gethostname().split('.')[0]
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        INSERT OR REPLACE INTO '{track_table}' 
        (track_id, clip_id, video_id, track_path, status, length, location, process_date_time) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (track_id, clip_id, video_id, str(track_path), Status.DOWNLOADED, int(length), location, datetime.now()))
    db.commit()
    db.close()
    if occlusion is not None:  # TODO: reduce no of database writes
        update_track(track_id, occlusion=occlusion, version=version, db_path=db_path)
    if flow is not None:
        update_track(track_id, flow=flow, version=version, db_path=db_path)
    if filter_reason is not None:
        update_track(track_id, status=Status.DISCARDED, filter_reason=filter_reason, version=version, db_path=db_path)


def insert_data(data_id, data_path, split, length, occlusion=None, flow=None, version=None, db_path=config_db_path()):
    data_version = parse_version(version).get("data")
    data_table = f"data_{data_version}" if data_version else "data"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        INSERT OR REPLACE INTO '{data_table}' (data_id, data_path, split, length, occlusion, flow, process_date_time) 
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """, (data_id, str(data_path), str(split), int(length), float(occlusion), float(flow), datetime.now()))
    db.commit()
    db.close()


def get_table(table_name):
    db = sqlite3.connect(config_db_path())
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT name FROM sqlite_master WHERE type = 'table' AND name = '{table_name}';
    """).fetchall()
    return convert_to_dicts(cursor, result)


def get_video(video_id, version=None, db_path=config_db_path()):
    """Get a video from video table"""
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT * from '{video_table}' WHERE video_id = '{video_id}';
    """).fetchall()
    result = convert_to_dicts(cursor, result)
    if result:
        return result[0]
    return None


def get_clip(clip_id, version=None, db_path=config_db_path()):
    """Get a clip from clip table"""
    clip_version = parse_version(version).get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT * from '{clip_table}' WHERE clip_id = '{clip_id}';
    """).fetchall()
    result = convert_to_dicts(cursor, result)
    if result:
        return result[0]
    return None


def get_track(track_id, version=None, db_path=config_db_path()):
    """Get a track from track table"""
    track_version = parse_version(version).get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT * from '{track_table}' WHERE track_id = '{track_id}';
    """).fetchall()
    result = convert_to_dicts(cursor, result)
    if result:
        return result[0]
    return {}


def get_data(data_id, version=None, db_path=config_db_path()):
    """Get a data from data table"""
    data_version = parse_version(version).get("data")
    data_table = f"data_{data_version}" if data_version else "data"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT * from '{data_table}' WHERE data_id = '{data_id}';
    """).fetchall()
    result = convert_to_dicts(cursor, result)
    if result:
        return result[0]
    return {}


def get_all_videos(condition=None, version=None, db_path=config_db_path()):
    """Get all videos in video table"""
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    if condition:
        result = cursor.execute(f"""
            SELECT * from '{video_table}' WHERE {condition};
        """).fetchall()
    else:
        result = cursor.execute(f"""
            SELECT * from '{video_table}';
        """).fetchall()
    return convert_to_dicts(cursor, result)


def get_all_clips(condition=None, version=None, db_path=config_db_path()):
    """Get all clips in clip table"""
    clip_version = parse_version(version).get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    if condition:
        result = cursor.execute(f"""
            SELECT * from '{clip_table}' WHERE {condition};
        """).fetchall()
    else:
        result = cursor.execute(f"""
            SELECT * from '{clip_table}';
        """).fetchall()
    return convert_to_dicts(cursor, result)


def get_all_tracks(condition=None, version=None, db_path=config_db_path()):
    """Get all tracks in track table"""
    track_version = parse_version(version).get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    if condition:
        result = cursor.execute(f"""
            SELECT * from '{track_table}' WHERE {condition};
        """).fetchall()
    else:
        result = cursor.execute(f"""
            SELECT * from '{track_table}';
        """).fetchall()
    return convert_to_dicts(cursor, result)


def get_all_data(version=None, db_path=config_db_path()):
    """Get all data in data table"""
    data_version = parse_version(version).get("data")
    data_table = f"data_{data_version}" if data_version else "data"
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    result = cursor.execute(f"""
        SELECT * from '{data_table}';
    """).fetchall()
    return convert_to_dicts(cursor, result)


def get_video_status(video_id, version=None, db_path=config_db_path()):
    video = get_video(video_id, version=version, db_path=db_path)
    if video is None:
        return None
    return video.get("status")


def get_clip_status(clip_id, version=None, db_path=config_db_path()):
    clip = get_clip(clip_id, version=version, db_path=db_path)
    if clip is None:
        return None
    return clip.get("status")


def update_video(video_id, version=None, db_path=config_db_path(), **kwargs):
    """Update a video in video table"""
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    for field, value in kwargs.items():
        if value is None or value.lower() == "null":
            value = "NULL"
        else:
            value = f"'{value}'"
        db.cursor().execute(f"""
            UPDATE '{video_table}' SET {field} = {value} WHERE video_id = '{video_id}';
        """)
    db.commit()
    db.close()


def update_clip(clip_id, version=None, db_path=config_db_path(), **kwargs):
    """Update a clip in clip table"""
    clip_version = parse_version(version).get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    db = sqlite3.connect(db_path)
    for field, value in kwargs.items():
        if value is None or value.lower() == "null":
            value = "NULL"
        else:
            value = f"'{value}'"
        db.cursor().execute(f"""
            UPDATE '{clip_table}' SET {field} = {value} WHERE clip_id = '{clip_id}';
        """)
    db.commit()
    db.close()


def update_track(track_id, version=None, db_path=config_db_path(), **kwargs):
    """Update a track in track table"""
    track_version = parse_version(version).get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    db = sqlite3.connect(db_path)
    for field, value in kwargs.items():
        if value is None:
            value = "NULL"
        elif isinstance(value, str):
            if value.lower() == "null":
                value = "NULL"
            else:
                value = f"'{value}'"
        else:  # float or int
            pass
        db.cursor().execute(f"""
            UPDATE '{track_table}' SET {field} = {value} WHERE track_id = '{track_id}';
        """)
    db.commit()
    db.close()


def update_video_status(video_id, status, version=None, db_path=config_db_path()):
    """Update the status of a video in video table"""
    update_video(video_id, status=status, version=version, db_path=db_path)


def update_video_path(video_id, video_path, version=None, db_path=config_db_path()):
    """Update the path of a video in video table"""
    update_video(video_id, video_path=str(video_path), version=version, db_path=db_path)


def update_track_status(track_id, status, version=None, db_path=config_db_path()):
    """Update the status of a track in track table"""
    update_track(track_id, status=status, version=version, db_path=db_path)
    # track_version = parse_version(version).get("track")
    # track_table = f"track_{track_version}" if track_version else "track"
    # db = sqlite3.connect(db_path)
    # curr_status = get_track(track_id, version=version, db_path=db_path).get("status")
    # if curr_status is None:
    #     db.close()
    #     raise ValueError(f"Fail to change status, {track_id} not in {track_table}")
    # db.cursor().execute(f"""
    #     UPDATE '{track_table}' SET status = '{status}' WHERE track_id = '{track_id}';
    # """)
    # db.commit()
    # db.close()


def update_track_field(track_id, field, value, version=None, db_path=config_db_path()):
    track_version = parse_version(version).get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    if field in ["occlusion", "flow"]:
        value = float(value)
    elif field in ["track_path", "location", "filter_reason"]:
        value = str(value)
    else:
        raise ValueError(f"Field {field} not in table or not mutable")
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        UPDATE '{track_table}' SET {field} = '{value}' WHERE track_id = '{track_id}';
    """)
    db.commit()
    db.close()


def update_data_status(data_id, status, version=None, db_path=config_db_path()):
    assert status in Status, f"Status {status} not valid"
    data_version = parse_version(version).get("data")
    data_table = f"data_{data_version}" if data_version else "data"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        UPDATE '{data_table}' SET feature_status = '{status}' WHERE data_id = '{data_id}';
    """)
    db.commit()
    db.close()


def update_data_field(data_id, field, value, version=None, db_path=config_db_path()):
    data_version = parse_version(version).get("data")
    data_table = f"data_{data_version}" if data_version else "data"
    if field == "status":
        value = str(value)
        assert value in Status, f"Status {value} not valid"
    elif field in ["data_path"]:
        value = str(value)
    else:
        raise ValueError(f"Field {field} not in table or not mutable")
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        UPDATE '{data_table}' SET {field} = '{value}' WHERE data_id = '{data_id}';
    """)
    db.commit()
    db.close()


def get_table_sum(table, column, condition=None, db_path=config_db_path()):
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    if condition:
        result = cursor.execute(f"""
            SELECT SUM({column}) FROM '{table}' WHERE {condition};
        """).fetchall()
    else:
        result = cursor.execute(f"""
            SELECT SUM({column}) FROM '{table}';
        """).fetchall()
    return result[0][0]


def discard_video(video_id, version=None, db_path=config_db_path()):
    """Change the status of a video to 'discarded'"""
    update_video(video_id, status=Status.DISCARDED, version=version, db_path=db_path)



def remove_video(video_id, version=None, db_path=config_db_path()):
    """Remove video from database and disk"""
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    video = get_video(video_id, version=version, db_path=db_path)
    if video is not None:
        video_path = video.get("video_path")
        if video_path and os.path.exists(video_path):
            print(f"Removing {video_path}", flush=True)
            try:
                os.remove(video_path)
            except:
                pass
        db = sqlite3.connect(db_path)
        db.cursor().execute(f"""
            DELETE FROM '{video_table}' WHERE video_id = '{video_id}';
        """)
        db.commit()
        db.close()


def discard_clip(video_id=None, clip_id=None, version=None, db_path=config_db_path()):
    """Change the status of clip(s) to 'discarded'"""
    version_dict = parse_version(version)
    clip_version = version_dict.get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    if video_id:
        db = sqlite3.connect(db_path)
        db.cursor().execute(f"""
            UPDATE '{clip_table}' SET status = '{Status.DISCARDED}' WHERE video_id = '{video_id}';
        """)
        db.commit()
        db.close()
    elif clip_id:
        update_clip(clip_id, status=Status.DISCARDED, version=version, db_path=db_path)


def remove_clip(video_id=None, clip_id=None, version=None, db_path=config_db_path()):
    """Remove clip(s) from clip table and disk by video_id (all clips of the video) or clip_id (one clip)"""
    version_dict = parse_version(version)
    clip_version = version_dict.get("clip")
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    all_clips = get_all_clips(condition="clip_path IS NOT NULL", version=version, db_path=db_path)
    if not all_clips:
        return
    clip_dir = all_clips[0].get("clip_path")
    clip_dir = Path(clip_dir).parent.parent.parent
    if video_id:
        clips = get_all_clips(f"video_id = '{video_id}'", version=version, db_path=db_path)
        for clip in clips:
            clip_path = clip.get("clip_path")
            if clip_path and os.path.exists(clip_path):
                print(f"Removing {clip_path}", flush=True)
                try:
                    os.remove(clip_path)
                except:
                    pass
        for cat_dir in list(clip_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for video_dir in list(cat_dir.iterdir()):
                if video_dir.name == video_id:
                    print(f"Removing {video_dir}", flush=True)
                    run(["rm", "-rf", video_dir])
        db = sqlite3.connect(db_path)
        db.cursor().execute(f"""
            DELETE FROM '{clip_table}' WHERE video_id = '{video_id}';
        """)
        db.commit()
        db.close()
    elif clip_id:
        clip = get_clip(clip_id, version=version, db_path=db_path)
        if clip is not None:
            clip_path = clip.get("clip_path")
            if clip_path and os.path.exists(clip_path):
                print(f"Removing {clip_path}", flush=True)
                try:
                    os.remove(clip_path)
                except:
                    pass
            db = sqlite3.connect(db_path)
            db.cursor().execute(f"""
                DELETE FROM '{clip_table}' WHERE clip_id = '{clip_id}';
            """)
            db.commit()
            db.close()



def remove_track(video_id=None, clip_id=None, track_id=None, version=None, db_path=config_db_path()):
    """
    Remove track(s) from track table and disk by
    video_id (all tracks of the video), clip_id (all tracks of the clip), or track_id (one track)
    """
    version_dict = parse_version(version)
    track_version = version_dict.get("track")
    track_table = f"track_{track_version}" if track_version else "track"
    all_tracks = get_all_tracks(
        condition="track_path IS NOT NULL AND location = 'viscam'",
        version=version, db_path=db_path
    )
    if not all_tracks:
        return
    track_dir = all_tracks[0].get("track_path")
    track_dir = Path(track_dir).parent.parent.parent
    if video_id:
        tracks = get_all_tracks(f"video_id = '{video_id}'", version=version, db_path=db_path)
        for track in tqdm(tracks, desc=f"Removing {video_id} tracks"):
            track_path = track.get("track_path")
            if track_path and Path(track_path).is_dir():
                print(f"Removing {track_path}", flush=True)
                run(["rm", "-rf", track_path])
        for cat_dir in list(track_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for clip_dir in list(cat_dir.iterdir()):
                if clip_dir.name.startswith(video_id):
                    print(f"Removing {clip_dir}", flush=True)
                    run(["rm", "-rf", clip_dir])
        db = sqlite3.connect(db_path)
        db.cursor().execute(f"""
            DELETE FROM '{track_table}' WHERE video_id = '{video_id}';
        """)
        db.commit()
        db.close()
    elif clip_id:
        tracks = get_all_tracks(f"clip_id = '{clip_id}'", version=version, db_path=db_path)
        for track in tracks:
            track_path = track.get("track_path")
            if track_path and Path(track_path).is_dir():
                print(f"Removing {track_path}", flush=True)
                run(["rm", "-rf", track_path])
        for cat_dir in list(track_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for clip_dir in list(cat_dir.iterdir()):
                if clip_dir.name == clip_id:
                    print(f"Removing {clip_dir}", flush=True)
                    run(["rm", "-rf", clip_dir])
        db = sqlite3.connect(db_path)
        db.cursor().execute(f"""
            DELETE FROM '{track_table}' WHERE clip_id = '{clip_id}';
        """)
        db.commit()
        db.close()
    elif track_id:
        track = get_track(track_id, version=version, db_path=db_path)
        if track is not None:
            track_path = track.get("track_path")
            if track_path and Path(track_path).is_dir():
                print(f"Removing {track_path}", flush=True)
                run(["rm", "-rfv", track_path])
            db = sqlite3.connect(db_path)
            db.cursor().execute(f"""
                DELETE FROM '{track_table}' WHERE track_id = '{track_id}';
            """)
            db.commit()
            db.close()


def drop_all_tables(version=None, db_path=config_db_path()):
    """Drop all tables of a version in database"""
    version_dict = parse_version(version)
    video_version = version_dict.get("video")
    clip_version = version_dict.get("clip")
    track_version = version_dict.get("track")
    video_table = f"video_{video_version}" if video_version else "video"
    clip_table = f"clip_{clip_version}" if clip_version else "clip"
    track_table = f"track_{track_version}" if track_version else "track"
    db = sqlite3.connect(db_path)
    for table_name in [video_table, clip_table, track_table]:
        db.cursor().execute(f"""
            DROP TABLE IF EXISTS '{table_name}';
        """)
    db.commit()
    db.close()


def rebuild_database(base_dir, version=None, db_path=config_db_path()):  # TODO refine
    """Rebuild database by iterating directories of video, clip, and track and insert them into database"""
    version_dict = parse_version(version)
    video_version = version_dict.get("video")
    clip_version = version_dict.get("clip")
    track_version = version_dict.get("track")
    drop_all_tables(version=version, db_path=db_path)  # TODO: add dataset table
    make_video_table(version=version, db_path=db_path)
    make_clip_table(version=version, db_path=db_path)
    make_track_table(version=version, db_path=db_path)
    video_dir = Path(base_dir) / f"video_{video_version}" if video_version else Path(base_dir) / "video"
    clip_dir = Path(base_dir) / f"clip_{clip_version}" if clip_version else Path(base_dir) / "clip"
    track_dir = Path(base_dir) / f"track_{track_version}" if track_version else Path(base_dir) / "track"
    if video_dir.is_dir():
        for cat_dir in video_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for video_path in tqdm(cat_dir.iterdir(), total=len(list(cat_dir.iterdir())), desc=f"Iterating {category} video"):
                if not video_path.suffix == ".mp4":
                    continue
                cap = cv2.VideoCapture(str(video_path))
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                insert_video(
                    video_id=video_path.stem, category=category, video_path=video_path, length=length,
                    query_text="", version=version, db_path=db_path
                )
                # TODO: fix arguments
                # TODO: also add discarded video
    else:
        print(f"Video directory {video_dir} not found", flush=True)
    if clip_dir.is_dir():
        for cat_dir in clip_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for video_dir in tqdm(cat_dir.iterdir(), total=len(list(cat_dir.iterdir())), desc=f"Iterating {category} clip"):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.stem
                for clip_path in video_dir.iterdir():
                    if clip_path.suffix != ".mp4":
                        continue
                    update_video_status(video_id=video_id, status=Status.PROCESSED, version=version, db_path=db_path)
                    cap = cv2.VideoCapture(str(clip_path))
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    clip_id = clip_path.stem
                    insert_clip(  # TODO: read start_time or start_frame from metadata
                        clip_id=clip_id, video_id=video_id, clip_path=clip_path, length=length, version=version,
                        db_path=db_path
                    )
                    # TODO: also add discarded clip
    else:
        print(f"Clip directory {clip_dir} not found", flush=True)
    if track_dir.is_dir():
        for cat_dir in track_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for clip_dir in tqdm(cat_dir.iterdir(), total=len(list(cat_dir.iterdir())), desc=f"Iterating {category} track"):
                if not clip_dir.is_dir():
                    continue
                clip_id = clip_dir.stem
                video_id = clip_id[:clip_id.rfind('_')]
                for track_dir in clip_dir.iterdir():
                    if not track_dir.is_dir() or clip_id not in track_dir.name:
                        continue
                    update_clip(clip_id=clip_id, status=Status.PROCESSED, version=version, db_path=db_path)
                    length = len([f for f in os.listdir(track_dir) if f.endswith("rgb.png")])
                    if length == 0:
                        continue
                    track_id = track_dir.stem
                    insert_track(
                        track_id=track_id, clip_id=clip_id, video_id=video_id, track_path=track_dir, length=length,
                        version=version, db_path=db_path
                    )
                    # TODO: read occlusion from metadata
                    # TODO: also add discarded track
    else:
        print(f"Track directory {track_dir} not found", flush=True)


def reset_video_status(status, version=None, db_path=config_db_path()):
    """Reset all video status to for processing of next stage"""
    video_version = parse_version(version).get("video")
    video_table = f"video_{video_version}" if video_version else "video"
    db = sqlite3.connect(db_path)
    db.cursor().execute(f"""
        UPDATE '{video_table}' SET status = '{status}';
    """)
    db.commit()
    db.close()


def retry(func):
    """Function decorator to retry on database lock error"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        retries = 0
        while retries < self.max_retry:
            try:
                return func(self, *args, **kwargs)
            except sqlite3.OperationalError as e:
                if retries > self.max_retry:
                    raise
                elif "locked" in str(e) or "busy" in str(e):
                    retries += 1
                    sleep(self.retry_sleep)
                else:
                    raise
    return wrapper


class Database:
    """Wrapper for database operations that hides db_path and version"""
    def __init__(self, db_path, version=None, max_retry=5, retry_sleep=3):
        self.db_path = db_path
        self.version = version
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep

    @retry  # TODO: maybe use class operator to add retry to all functions
    def insert_video(
        self, video_id, category, query_text, fps=None, duration=None, frames=None, title=None, keywords=None,
        video_path=None, reason=None
    ):
        insert_video(
            video_id=video_id, category=category, video_path=video_path, fps=fps, duration=duration, frames=frames,
            title=title, keywords=keywords, query_text=query_text, reason=reason,
            version=self.version, db_path=self.db_path
        )

    @retry
    def insert_clip(self, clip_id, video_id, clip_path, start_time, start_frame, duration, frames, filter_reason=None):
        insert_clip(
            clip_id, video_id, clip_path, start_time, start_frame, duration, frames, filter_reason,
            self.version, self.db_path
        )

    @retry
    def insert_track(
        self, track_id, clip_id, video_id, track_path, length, occlusion=None, flow=None, filter_reason=None
    ):
        insert_track(
            track_id, clip_id, video_id, track_path, length, occlusion, flow, filter_reason, self.version, self.db_path
        )

    @retry
    def insert_data(self, data_id, data_path, split, length, occlusion=None, flow=None):
        insert_data(data_id, data_path, split, length, occlusion, flow, self.version, self.db_path)

    @retry
    def get_video(self, video_id):
        return get_video(video_id, self.version, self.db_path)

    @retry
    def get_clip(self, clip_id):
        return get_clip(clip_id, self.version, self.db_path)

    @retry
    def get_track(self, track_id):
        return get_track(track_id, self.version, self.db_path)

    @retry
    def get_data(self, data_id):
        return get_data(data_id, self.version, self.db_path)

    @retry
    def get_all_videos(self, condition=None):
        return get_all_videos(condition, self.version, self.db_path)

    @retry
    def get_all_clips(self, condition=None):
        return get_all_clips(condition, self.version, self.db_path)

    @retry
    def get_all_tracks(self, condition=None):
        return get_all_tracks(condition, self.version, self.db_path)

    @retry
    def get_all_data(self):
        return get_all_data(self.version, self.db_path)

    @retry
    def get_video_status(self, video_id):
        return get_video_status(video_id, self.version, self.db_path)

    @retry
    def get_clip_status(self, clip_id):
        return get_clip_status(clip_id, self.version, self.db_path)

    @retry
    def update_video(self, video_id, **kwargs):
        update_video(video_id, version=self.version, db_path=self.db_path, **kwargs)

    @retry
    def update_clip(self, clip_id, **kwargs):
        update_clip(clip_id, version=self.version, db_path=self.db_path, **kwargs)

    @retry
    def update_track(self, track_id, **kwargs):
        update_track(track_id, version=self.version, db_path=self.db_path, **kwargs)

    @retry
    def update_video_status(self, video_id, status):
        update_video_status(video_id, status, self.version, self.db_path)

    @retry
    def update_track_status(self, track_id, status):
        update_track_status(track_id, status, self.version, self.db_path)

    @retry
    def update_data_status(self, data_id, status):
        update_data_status(data_id, status, self.version, self.db_path)

    @retry
    def update_track_field(self, track_id, field, value):
        update_track_field(track_id, field, value, self.version, self.db_path)

    @retry
    def remove_video(self, video_id):
        remove_video(video_id, version=self.version, db_path=self.db_path)

    @retry
    def remove_clip(self, video_id=None, clip_id=None):
        remove_clip(video_id=video_id, clip_id=clip_id, version=self.version, db_path=self.db_path)

    @retry
    def remove_track(self, video_id=None, clip_id=None, track_id=None):
        remove_track(video_id=video_id, clip_id=clip_id, track_id=track_id, version=self.version, db_path=self.db_path)

    @retry
    def discard_video(self, video_id):
        discard_video(video_id, version=self.version, db_path=self.db_path)

    @retry
    def discard_clip(self, video_id=None, clip_id=None):
        discard_clip(video_id=video_id, clip_id=clip_id, version=self.version, db_path=self.db_path)

    @retry
    def make_video_table(self):
        make_video_table(self.version, self.db_path)

    @retry
    def make_clip_table(self):
        make_clip_table(self.version, self.db_path)

    @retry
    def make_track_table(self):
        make_track_table(self.version, self.db_path)

    @retry
    def make_data_table(self):
        make_data_table(self.version, self.db_path)

