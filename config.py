"""Central configuration for the AI Based Intelligent Traffic Management System."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "traffic.db")

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-key")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))

# ---- AI models -------------------------------------------------------------
# Vehicle detection: pretrained YOLOv8 (COCO). Auto-downloads on first run.
VEHICLE_MODEL = os.environ.get("VEHICLE_MODEL", "yolov8n.pt")
# COCO class ids: 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Emergency vehicle detection: a SECOND pretrained model (see README).
# Put your downloaded weights at models/emergency.pt (or set EMERGENCY_MODEL).
EMERGENCY_MODEL = os.environ.get("EMERGENCY_MODEL", os.path.join(MODEL_DIR, "emergency.pt"))
# Class names that count as "emergency". If the model has none of these names,
# every class of that model is treated as emergency.
EMERGENCY_KEYWORDS = ("ambulance", "fire", "police", "emergency", "siren", "rescue")

# ---- Lanes -----------------------------------------------------------------
LANE_NAMES = ["Lane 1 (North)", "Lane 2 (East)", "Lane 3 (South)", "Lane 4 (West)"]
NUM_LANES = len(LANE_NAMES)
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
MAX_UPLOAD_MB = 800

# ---- Default admin-editable settings (stored in SQLite) ---------------------
DEFAULT_SETTINGS = {
    "min_green": 10,              # seconds of green with zero vehicles
    "max_green": 60,              # upper limit of green time
    "seconds_per_vehicle": 2.0,   # extra green seconds per detected vehicle
    "yellow_time": 3,             # yellow seconds before switching lane
    "vehicle_conf": 0.35,         # YOLOv8 vehicle confidence threshold
    "emergency_conf": 0.50,       # emergency model confidence threshold
    "emergency_clear_delay": 4,   # seconds without emergency before releasing
}
