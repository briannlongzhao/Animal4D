from flask import Flask, render_template, request, redirect, url_for, jsonify
import database as db


app = Flask(__name__, template_folder='.', static_folder=None)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/browse_video", methods=["GET"])
def browse_video():
    videos = db.get_all_videos()
    return render_template("browse_video.html", videos=videos)


@app.route('/delete_video/<video_id>', methods=['POST'])
def delete_video(video_id):
    db.remove_video(video_id)
    return redirect(url_for('browse_video'))
#
# @app.route('/delete/<int:id>', methods=['POST'])
# def delete_video(id):
#     conn = get_db_connection()
#     conn.execute('DELETE FROM videos WHERE id = ?', (id,))
#     conn.commit()
#     conn.close()
#     return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)