"""Lane video readers + AI detection engine (YOLOv8 vehicles + emergency model)."""
import os
import threading
import time
from collections import deque

import config

FRAME_WIDTH = 640


class Lane:
    def __init__(self, idx, name):
        self.idx = idx
        self.name = name
        self.video_path = None
        self.lock = threading.Lock()
        self.frame = None            # latest resized frame (numpy)
        self.boxes = []              # [x1, y1, x2, y2, label, is_emergency, conf]
        self.count = 0
        self.detected_emergency = False
        self.force_emergency = False  # admin "simulate emergency" switch
        self.emergency_since = None
        self.status = "No video uploaded"
        self.signal = "RED"
        self._em_hist = deque(maxlen=5)
        self._last_em_seen = 0.0
        self._stop = threading.Event()
        self._thread = None

    @property
    def emergency(self):
        return self.detected_emergency or self.force_emergency

    def set_force(self, value):
        if value and not self.emergency:
            self.emergency_since = time.time()
        self.force_emergency = value

    # ------------------------------------------------------------ video reading
    def set_video(self, path):
        self.clear_video()
        self.video_path = path
        self._stop = threading.Event()
        self.status = "Starting video..."
        self._thread = threading.Thread(target=self._read_loop, args=(self._stop, path),
                                        daemon=True, name=f"lane{self.idx}-reader")
        self._thread.start()

    def clear_video(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        self.video_path = None
        with self.lock:
            self.frame, self.boxes, self.count = None, [], 0
        self.detected_emergency = False
        self._em_hist.clear()
        self.status = "No video uploaded"

    def _read_loop(self, stop, path):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status = "Cannot open video file"
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps and 1 < fps < 120 else 1 / 25
        fails = 0
        self.status = "Running"
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 3:
                    self.status = "Video unreadable"
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                continue
            fails = 0
            h, w = frame.shape[:2]
            if w != FRAME_WIDTH:
                frame = cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))
            with self.lock:
                self.frame = frame
            time.sleep(delay)
        cap.release()

    def snapshot_frame(self):
        with self.lock:
            return (None if self.frame is None else self.frame.copy()), list(self.boxes)


class DetectionEngine:
    """Round-robin inference over all lanes using two models."""

    def __init__(self, lanes, get_settings, log):
        self.lanes = lanes
        self.get_settings = get_settings
        self.log = log
        self.vehicle_model = None
        self.em_model = None
        self.em_class_ids = set()
        self.status = "Loading models..."
        self.emergency_mode = "none"   # model | heuristic
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="detector")
        self._thread.start()

    def stop(self):
        self._stop.set()

    # --------------------------------------------------------------- model load
    def _load_models(self):
        from ultralytics import YOLO
        self.vehicle_model = YOLO(config.VEHICLE_MODEL)
        if os.path.exists(config.EMERGENCY_MODEL):
            self.em_model = YOLO(config.EMERGENCY_MODEL)
            names = self.em_model.names
            ids = {i for i, n in names.items() if any(k in str(n).lower() for k in config.EMERGENCY_KEYWORDS)}
            self.em_class_ids = ids or set(names.keys())
            self.emergency_mode = "model"
            self.status = f"Ready: {config.VEHICLE_MODEL} + {os.path.basename(config.EMERGENCY_MODEL)}"
        else:
            self.emergency_mode = "heuristic"
            self.status = ("Ready: vehicle model loaded. models/emergency.pt not found - using "
                           "siren-light colour heuristic (demo only). See README.")

    # ---------------------------------------------------------------- inference
    def _detect_vehicles(self, frame, conf):
        res = self.vehicle_model.predict(frame, imgsz=640, conf=conf,
                                         classes=config.VEHICLE_CLASS_IDS, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append([x1, y1, x2, y2, res.names[int(b.cls[0])], False, float(b.conf[0])])
        return out

    def _detect_emergency_model(self, frame, conf):
        res = self.em_model.predict(frame, imgsz=640, conf=conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls[0])
            if cid in self.em_class_ids:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                out.append([x1, y1, x2, y2, str(res.names[cid]), True, float(b.conf[0])])
        return out

    @staticmethod
    def _detect_emergency_heuristic(frame, vehicle_boxes):
        """Fallback: red + blue siren-light pixels on the upper part of a vehicle box."""
        import cv2
        out = []
        for x1, y1, x2, y2, label, _, conf in vehicle_boxes:
            if label == "motorcycle":
                continue
            top = frame[max(0, y1): max(1, y1 + int((y2 - y1) * 0.45)), max(0, x1): max(1, x2)]
            if top.size == 0:
                continue
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            bright = (s > 150) & (v > 200)
            red = bright & ((h < 8) | (h > 172))
            blue = bright & (h > 100) & (h < 130)
            area = top.shape[0] * top.shape[1]
            if red.sum() / area > 0.006 and blue.sum() / area > 0.006:
                out.append([x1, y1, x2, y2, "emergency (lights)", True, 0.5])
        return out

    # ------------------------------------------------------------------ main loop
    def _run(self):
        try:
            self._load_models()
        except Exception as exc:
            self.status = f"Model load failed: {exc}"
            print(self.status)
            return
        while not self._stop.is_set():
            worked = False
            for lane in self.lanes:
                if self._stop.is_set():
                    break
                frame, _ = lane.snapshot_frame()
                if frame is None:
                    continue
                worked = True
                try:
                    self._process(lane, frame)
                except Exception as exc:
                    print(f"detect error lane {lane.idx}:", exc)
            if not worked:
                time.sleep(0.2)

    def _process(self, lane, frame):
        s = self.get_settings()
        vehicles = self._detect_vehicles(frame, s["vehicle_conf"])
        if self.emergency_mode == "model":
            em_boxes = self._detect_emergency_model(frame, s["emergency_conf"])
        else:
            em_boxes = self._detect_emergency_heuristic(frame, vehicles)

        now = time.time()
        lane._em_hist.append(bool(em_boxes))
        if em_boxes:
            lane._last_em_seen = now
        if not lane.detected_emergency and sum(lane._em_hist) >= 2:   # confirm over frames
            lane.detected_emergency = True
            lane.emergency_since = now
            self.log(lane.idx, "detect", f"Emergency vehicle detected in {lane.name}")
        elif lane.detected_emergency and now - lane._last_em_seen > s["emergency_clear_delay"]:
            lane.detected_emergency = False
            self.log(lane.idx, "detect", f"Emergency vehicle left {lane.name}")

        # mark vehicle boxes overlapped by an emergency box as emergency, add model boxes
        boxes = [v for v in vehicles if not any(_iou(v, e) > 0.4 for e in em_boxes)] + em_boxes
        with lane.lock:
            lane.boxes = boxes
            lane.count = len(vehicles) if not em_boxes else max(len(vehicles), len(boxes))


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
