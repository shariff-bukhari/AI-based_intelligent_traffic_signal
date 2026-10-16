"""AI Based Intelligent Traffic Management System - Flask web application.

Run:  python app.py   ->  http://127.0.0.1:5000   (default admin: admin / admin123)
"""
import glob
import os
import time
from functools import wraps

from flask import (Flask, Response, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import config
import db
from controller import SignalController
from streams import DetectionEngine, Lane

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
db.init_db()

lanes = [Lane(i, n) for i, n in enumerate(config.LANE_NAMES)]
controller = SignalController(lanes, db.get_settings, db.log_event)
engine = DetectionEngine(lanes, db.get_settings, db.log_event)


def lane_video_files(i):
    return glob.glob(os.path.join(config.UPLOAD_DIR, f"lane{i}.*"))


def load_existing_videos():
    for i, lane in enumerate(lanes):
        files = lane_video_files(i)
        if files:
            lane.set_video(files[0])


# ------------------------------------------------------------------ auth helpers
def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if "uid" not in session:
            if request.path.startswith("/api/") or request.path.startswith("/video_feed"):
                return jsonify(error="login required"), 401
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper


def admin_required(f):
    @wraps(f)
    @login_required
    def wrapper(*a, **kw):
        if session.get("role") != "admin":
            flash("Admin access required.", "error")
            return redirect(url_for("dashboard"))
        return f(*a, **kw)
    return wrapper


# ------------------------------------------------------------------------ pages
@app.route("/")
def index():
    return redirect(url_for("dashboard" if "uid" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        with db.db() as c:
            row = c.execute("SELECT * FROM users WHERE username=?", (u,)).fetchone()
        if row and check_password_hash(row["password_hash"], p):
            session.clear()
            session.update(uid=row["id"], username=row["username"], role=row["role"])
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", lanes=lanes)


# --------------------------------------------------------------------- live data
def placeholder_jpeg(text):
    import cv2
    import numpy as np
    img = np.full((360, 640, 3), 28, np.uint8)
    cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def mjpeg(lane):
    import cv2
    colours = {"GREEN": (80, 200, 60), "YELLOW": (0, 200, 240), "RED": (60, 60, 230)}
    while True:
        frame, boxes = lane.snapshot_frame()
        if frame is None:
            data = placeholder_jpeg(lane.status)
            time.sleep(0.5)
        else:
            for x1, y1, x2, y2, label, is_em, conf in boxes:
                col = (0, 0, 255) if is_em else (255, 160, 40)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if is_em else 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(frame, f"{lane.name} | vehicles: {lane.count} | {lane.signal}", (8, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours.get(lane.signal, (255, 255, 255)), 2)
            data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            time.sleep(0.04)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"


@app.route("/video_feed/<int:i>")
@login_required
def video_feed(i):
    if not 0 <= i < len(lanes):
        return "Not found", 404
    return Response(mjpeg(lanes[i]), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
@login_required
def api_state():
    ctl = controller.snapshot()
    for lane, sig in zip(lanes, ctl["signals"]):
        lane.signal = sig
    return jsonify(
        controller=ctl,
        engine={"status": engine.status, "emergency_mode": engine.emergency_mode},
        lanes=[{
            "idx": l.idx, "name": l.name, "count": l.count, "signal": ctl["signals"][l.idx],
            "emergency": l.emergency, "forced": l.force_emergency,
            "has_video": l.video_path is not None, "status": l.status,
        } for l in lanes],
        events=db.recent_events(12),
        now=time.time(),
    )


# ------------------------------------------------------------------------- admin
@app.route("/admin")
@admin_required
def admin():
    with db.db() as c:
        users = c.execute("SELECT id,username,role FROM users ORDER BY id").fetchall()
    return render_template("admin.html", lanes=lanes, settings=db.get_settings(), users=users,
                           events=db.recent_events(60), engine=engine,
                           emergency_model_path=config.EMERGENCY_MODEL,
                           config_ext=", ".join(sorted(config.ALLOWED_VIDEO_EXT)))


@app.route("/admin/upload/<int:i>", methods=["POST"])
@admin_required
def admin_upload(i):
    f = request.files.get("video")
    if not 0 <= i < len(lanes) or not f or not f.filename:
        flash("Choose a video file first.", "error")
        return redirect(url_for("admin"))
    ext = secure_filename(f.filename).rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in config.ALLOWED_VIDEO_EXT:
        flash(f"Unsupported file type. Allowed: {', '.join(sorted(config.ALLOWED_VIDEO_EXT))}", "error")
        return redirect(url_for("admin"))
    lanes[i].clear_video()
    for old in lane_video_files(i):
        os.remove(old)
    path = os.path.join(config.UPLOAD_DIR, f"lane{i}.{ext}")
    f.save(path)
    lanes[i].set_video(path)
    db.log_event(i, "admin", f"{session['username']} uploaded video for {lanes[i].name}")
    flash(f"Video uploaded for {lanes[i].name}.", "ok")
    return redirect(url_for("admin"))


@app.route("/admin/remove/<int:i>", methods=["POST"])
@admin_required
def admin_remove(i):
    if 0 <= i < len(lanes):
        lanes[i].clear_video()
        for old in lane_video_files(i):
            os.remove(old)
        flash(f"Video removed from {lanes[i].name}.", "ok")
    return redirect(url_for("admin"))


@app.route("/admin/simulate/<int:i>", methods=["POST"])
@admin_required
def admin_simulate(i):
    if 0 <= i < len(lanes):
        lanes[i].set_force(not lanes[i].force_emergency)
        state = "ON" if lanes[i].force_emergency else "OFF"
        db.log_event(i, "admin", f"Simulated emergency {state} for {lanes[i].name} by {session['username']}")
    return redirect(url_for("admin"))


@app.route("/admin/settings", methods=["POST"])
@admin_required
def admin_settings():
    new, limits = {}, {
        "min_green": (3, 120), "max_green": (5, 300), "seconds_per_vehicle": (0, 20),
        "yellow_time": (1, 10), "vehicle_conf": (0.05, 0.95), "emergency_conf": (0.05, 0.95),
        "emergency_clear_delay": (1, 30),
    }
    try:
        for k, (lo, hi) in limits.items():
            v = float(request.form[k])
            if not lo <= v <= hi:
                raise ValueError(f"{k} must be between {lo} and {hi}")
            new[k] = v
        if new["max_green"] < new["min_green"]:
            raise ValueError("max_green must be >= min_green")
    except (ValueError, KeyError) as e:
        flash(f"Invalid settings: {e}", "error")
        return redirect(url_for("admin"))
    db.save_settings(new)
    flash("Settings saved.", "ok")
    return redirect(url_for("admin"))


@app.route("/admin/users/add", methods=["POST"])
@admin_required
def admin_user_add():
    u = request.form.get("username", "").strip()
    p = request.form.get("password", "")
    role = request.form.get("role", "viewer")
    if not u or len(p) < 6 or role not in ("admin", "viewer"):
        flash("Username required and password must be at least 6 characters.", "error")
        return redirect(url_for("admin"))
    try:
        with db.db() as c:
            c.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                      (u, generate_password_hash(p), role))
        flash(f"User {u} created.", "ok")
    except Exception:
        flash("That username already exists.", "error")
    return redirect(url_for("admin"))


@app.route("/admin/users/<int:uid>/delete", methods=["POST"])
@admin_required
def admin_user_delete(uid):
    if uid == session["uid"]:
        flash("You cannot delete your own account.", "error")
    else:
        with db.db() as c:
            c.execute("DELETE FROM users WHERE id=?", (uid,))
        flash("User deleted.", "ok")
    return redirect(url_for("admin"))


@app.route("/admin/password", methods=["POST"])
@login_required
def change_password():
    p = request.form.get("password", "")
    if len(p) < 6:
        flash("Password must be at least 6 characters.", "error")
    else:
        with db.db() as c:
            c.execute("UPDATE users SET password_hash=? WHERE id=?",
                      (generate_password_hash(p), session["uid"]))
        flash("Password updated.", "ok")
    return redirect(url_for("admin" if session.get("role") == "admin" else "dashboard"))


def start_background():
    load_existing_videos()
    controller.start()
    engine.start()


if __name__ == "__main__":
    start_background()
    print(f"\n  Traffic AI running at http://{config.HOST}:{config.PORT}  (admin / admin123)\n")
    app.run(host=config.HOST, port=config.PORT, threaded=True, debug=False, use_reloader=False)
