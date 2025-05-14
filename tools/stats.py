import os
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import database as db


spreadsheet_id = "1FWpcSpgpfUe2xBIZgStqN72_nP_TKGzlCvaaIZgPJJg"
# spreadsheet_id = "1TXzOQUeo2mVQoub0bypne-4Prriw9W3OL6k1lpdzWfw"  # test sheet
version = "2.0.0"
db_path = "data/database.sqlite"
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
fps = 10


print("Authorize and connect to Google Sheets API", flush=True)
creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", scopes)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
        creds = flow.run_local_server(port=0)
    with open("token.json", "w") as token:
        token.write(creds.to_json())
spreadsheet = build("sheets", "v4", credentials=creds).spreadsheets()
metadata = spreadsheet.get(spreadsheetId=spreadsheet_id).execute()


print("Add main sheet if not exist", flush=True)
main_sheet_exist = False
main_sheet_id = None
for sheet in metadata["sheets"]:
    if sheet["properties"]["title"] == "main":
        main_sheet_exist = True
        main_sheet_id = sheet["properties"]["sheetId"]
        break
if not main_sheet_exist:
    body = {"requests": [{"addSheet": {"properties": {"title": "main"}}}]}
    response = spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
    main_sheet_id = response["replies"][0]["addSheet"]["properties"]["sheetId"]


print("Update main sheet title row", flush=True)
values = [
    "date", "version", "video", "video frame", "video hour", "clip", "clip frame", "clip hour", "track", "track frame",
    "track hour", "track/clip", "track/video", "clip/video", "track yield", "track frame yield", "clip frame yield",
    "video frame yield", "yield track frame", "yield track hour"
]
notes = [
    "", (  # date
        "{video_version}_{clip_version}_{track_version}\nvideo_version: downloaded raw video in stage 1\nclip_version: "
        "processed clips after stage 2\ntrack_version: processed tracks after stage 3"  # version
    ), "", "", "", "", "", "", "", "", "",
    # video, video frame, video hour, clip, clip frame, clip hour, track, track frame, track hour
    "# of processed track frames / # of clip frames",  # track/clip
    "# of processed track frames / # of video frames",  # track/video
    "# of processed clip frames / # of video frames",  # clip/video
    "(approximate) # of acceptable tracks / # of total processed tracks",  # track yield
    "(approximate) # of acceptable track frames / # of total processed track frames",  # track frame yield
    "(approximate) # of acceptable track frames / # of total clip frames",  # clip frame yield
    "(approximate) # of acceptable track frames / # of total video frames",  # video frame yield
    "(approximate) acceptable track frames",  # yield track frame
    "(approximate) acceptable track hours",  # yield track hour
]
assert len(values) == len(notes), f"len(values) {len(values)} != len(notes) {len(notes)}"
base_freeze_row_request = {"updateSheetProperties": {
    "properties": {"sheetId": main_sheet_id, "gridProperties": {"frozenRowCount": 1}},
    "fields": "gridProperties.frozenRowCount"
}}
base_update_cell_request = {"updateCells": {
    "start": {"sheetId": main_sheet_id, "rowIndex": 0, "columnIndex": 0},
    "rows": {"values": []},
    "fields": "*"
}}
base_auto_resize_request = {"autoResizeDimensions": {
    "dimensions": {"sheetId": main_sheet_id, "dimension": "COLUMNS"}}
}
body = {"requests": [base_freeze_row_request]}
update_shell_request = deepcopy(base_update_cell_request)
for value, note in zip(values, notes):
    update_shell_request["updateCells"]["rows"]["values"].append({
        "userEnteredValue": {"stringValue": value},
        "note": note
    })
body["requests"].append(update_shell_request)
body["requests"].append(base_auto_resize_request)
spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


print("Remove existing version sheet if exist", flush=True)
for sheet in metadata["sheets"]:
    if sheet["properties"]["title"] == version:
        sheet_id = sheet["properties"]["sheetId"]
        body = {"requests": [{"deleteSheet": {"sheetId": sheet_id}}]}
        spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


print("Append a new version sheet", flush=True)
body = {"requests": [{"addSheet": {"properties": {"title": version}}}]}
response = spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
sheet_id = response["replies"][0]["addSheet"]["properties"]["sheetId"]


print("Add title row to the version sheet", flush=True)
colum_names = [
    "category", "video", "video frame", "clip", "clip frame", "track", "track frame", "occlusion", "flow", "acceptable"
]
body = {"values": [colum_names]}
spreadsheet.values().update(
    spreadsheetId=spreadsheet_id, range=f"{version}!A1", valueInputOption="RAW", body=body
).execute()
freeze_row_request = deepcopy(base_freeze_row_request)
freeze_row_request["updateSheetProperties"]["properties"]["sheetId"] = sheet_id
body = {"requests": [freeze_row_request]}
spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


print("Add data rows to the version sheet", flush=True)
values = []
base_url = """=HYPERLINK("vcv.stanford.edu/cgi-bin/file-explorer/?dir="""
display_option = "&patterns_show={}&patterns_highlight=&autoplay=1&showmedia=1"
all_tracks = db.get_all_tracks(version=version, db_path=db_path)
for count, track in tqdm(enumerate(all_tracks), total=len(all_tracks)):
    if track.get("filter_reason") is not None:
        continue
    video_id = track.get("video_id")
    clip_id = track.get("clip_id")
    track_id = track.get("track_id")
    video_path = Path(db.get_video(video_id=video_id, version=version, db_path=db_path).get("video_path"))
    clip_path = Path(db.get_clip(clip_id=clip_id, version=version, db_path=db_path).get("clip_path"))
    track_path = Path(track.get("track_path"))
    video_frame = db.get_video(video_id=video_id, version=version, db_path=db_path).get("length")
    clip_frame = db.get_clip(clip_id=clip_id, version=version, db_path=db_path).get("length")
    track_frame = track.get("length")
    occlusion = track.get("occlusion")
    flow = track.get("flow")
    video_text = (
        base_url + f"""{video_path.parent.resolve()}{display_option.format(video_path.stem+'*')}", "{video_id}")"""
    )
    clip_text = (
        base_url + f"""{clip_path.parent.resolve()}{display_option.format(clip_path.name)}", "{clip_id}")"""
    )
    track_text = (
        base_url + f"""{Path(track_path).resolve()}{display_option.format("*.mp4")}", "{track_id}")"""
    )
    category = db.get_video(video_id=video_id, version=version, db_path=db_path).get("category")
    values.append([category, video_text, video_frame, clip_text, clip_frame, track_text, track_frame, occlusion, flow])
    if count % 1000 == 0 or count == len(all_tracks) - 1:  # batch update every 1000 rows
        body = {"values": values}
        spreadsheet.values().append(
            spreadsheetId=spreadsheet_id, range=f"{version}!A1", valueInputOption="USER_ENTERED", body=body
        ).execute()
        values = []


print("Add acceptable dropdown column", flush=True)
set_validation_request = {"setDataValidation": {
    "range": {
        "sheetId": sheet_id,
        "startRowIndex": 1,
        "endRowIndex": len(values) + 1,
        "startColumnIndex": len(colum_names) - 1,
        "endColumnIndex": len(colum_names)
    },
    "rule": {
        "condition": {"type": "ONE_OF_LIST", "values": [{"userEnteredValue": "yes"}, {"userEnteredValue": "no"}]},
        "strict": True,
        "showCustomUi": True
    }
}}
base_format_request = {"addConditionalFormatRule": {"rule": {
    "ranges": [{
        "sheetId": sheet_id,
        "startRowIndex": 1,
        "endRowIndex": len(values) + 1,
        "startColumnIndex": len(colum_names) - 1,
        "endColumnIndex": len(colum_names)
    }],
    "booleanRule": {
        "condition": {"type": "TEXT_EQ", "values": [{"userEnteredValue": ""}]},
        "format": {"backgroundColorStyle": {"rgbColor": {}}}
    }
}}}
yes_format_request = deepcopy(base_format_request)
yes_format_request["addConditionalFormatRule"]["rule"]["booleanRule"]["condition"]["values"][0]["userEnteredValue"] = \
    "yes"
yes_format_request["addConditionalFormatRule"]["rule"]["booleanRule"]["format"]["backgroundColorStyle"]["rgbColor"] = \
    {"red": 0.831, "green": 0.929, "blue": 0.737}
no_format_request = deepcopy(base_format_request)
no_format_request["addConditionalFormatRule"]["rule"]["booleanRule"]["condition"]["values"][0]["userEnteredValue"] = \
    "no"
no_format_request["addConditionalFormatRule"]["rule"]["booleanRule"]["format"]["backgroundColorStyle"]["rgbColor"] = \
    {"red": 1, "green": 0.812, "blue": 0.788}
auto_resize_request = deepcopy(base_auto_resize_request)
auto_resize_request["autoResizeDimensions"]["dimensions"]["sheetId"] = sheet_id
body = {"requests": [set_validation_request, no_format_request, yes_format_request, auto_resize_request]}
spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


print("Append row to main sheet", flush=True)
version_dict = db.parse_version(version)
video_version = version_dict.get("video")
clip_version = version_dict.get("clip")
track_version = version_dict.get("track")
video_table = f"video_{video_version}" if video_version else "video"
clip_table = f"clip_{clip_version}" if clip_version else "clip"
track_table = f"track_{track_version}" if track_version else "track"
video = len(db.get_all_videos(condition=f"status = '{db.Status.PROCESSED}'", version=version, db_path=db_path))
video_frame = db.get_table_sum(
    table=video_table, column="length", condition=f"status='{db.Status.PROCESSED}'", db_path=db_path
)
video_hour = f"{video_frame/fps/3600:.3f}"
clip = len(db.get_all_clips(condition=f"status='{db.Status.PROCESSED}'", version=version, db_path=db_path))
clip_frame = db.get_table_sum(
    table=clip_table, column="length", condition=f"status='{db.Status.PROCESSED}'", db_path=db_path
)
clip_hour = f"{clip_frame/fps/3600:.3f}"
track = len(db.get_all_tracks(version=version, db_path=db_path))
track_frame = db.get_table_sum(table=track_table, column="length", db_path=db_path)
track_hour = f"{track_frame/fps/3600:.3f}"
track_frame_clip_frame = f"{track_frame/clip_frame:.3f}" if clip_frame else "#DIV/0"
track_frame_video_frame = f"{track_frame/video_frame:.3f}" if video_frame else "#DIV/0"
clip_frame_video_frame = f"{clip_frame/video_frame:.3f}" if video_frame else "#DIV/0"
acceptable_track = f"""COUNTIF('{version}'!J2:J, "yes")"""
ann_track = f"""COUNTIF('{version}'!J2:J, "<>")"""
acceptable_frame = f"""SUMIF('{version}'!J2:J, "yes", '{version}'!G2:G)"""
ann_track_frame = f"""SUMIF('{version}'!J2:J, "<>", '{version}'!G2:G)"""
ann_clip = f"""COUNTA(UNIQUE(FILTER('{version}'!D2:D, '{version}'!J2:J <> "")))"""
ann_clip_frame = (
    f"""SUM(UNIQUE(FILTER('{version}'!E2:E, '{version}'!J2:I <> "", MATCH('{version}'!D2:D, """
    f"""UNIQUE(FILTER('{version}'!D2:D, '{version}'!J2:I <> "")), 0))))"""
)
ann_video = f"""COUNTA(UNIQUE(FILTER('{version}'!B2:B, '{version}'!J2:I <> "")))"""
ann_video_frame = (
    f"""SUM(UNIQUE(FILTER('{version}'!C2:C, '{version}'!J2:I <> "", MATCH('{version}'!B2:B, """
    f"""UNIQUE(FILTER('{version}'!B2:B, '{version}'!J2:I <> "")), 0))))"""
)
body = {"values": [[
    datetime.now().strftime("%Y-%m-%d"), version,
    video, video_frame, video_hour, clip, clip_frame, clip_hour, track, track_frame, track_hour,
    track_frame_clip_frame, track_frame_video_frame, clip_frame_video_frame,
    f"={acceptable_track}/{ann_track}",
    f"={acceptable_frame}/{ann_track_frame}",
    f"={acceptable_frame}/{ann_clip_frame}",
    f"={acceptable_frame}/{ann_video_frame}",
    f"=J:J * O:O", f"=K:K * O:O"
]]}
spreadsheet.values().append(
    spreadsheetId=spreadsheet_id, range=f"main!A1", valueInputOption="USER_ENTERED", body=body
).execute()
body = {"requests": [base_auto_resize_request]}
spreadsheet.batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()


