from random import shuffle
from database import Database
from traceback import print_exc


"Constantly remove the GPT filtered data from the database and disk"


version = "3.0.0"
db_path = "data/database.sqlite"

if __name__ == "__main__":
    db = Database(version=version, db_path=db_path)
    while True:
        all_tracks = db.get_all_tracks(condition="gpt_filtered = False")
        shuffle(all_tracks)
        for track in all_tracks:
            try:
                assert track.get("gpt_filtered") == False
                track_id = track.get("track_id")
                db.remove_track(track_id=track_id)
            except:
                print_exc()
                continue
