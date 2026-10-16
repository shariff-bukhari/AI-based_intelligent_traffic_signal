# AI Based Intelligent Traffic Management System

A 4-lane smart junction simulator that runs on uploaded videos.

* **YOLOv8** (pretrained COCO) counts cars, motorcycles, buses and trucks in each lane.
* **Green time per lane** = `min_green + seconds_per_vehicle × vehicles` (capped at `max_green`).
* Lanes get green in **cyclic order** 1 → 2 → 3 → 4 → 1 … with a yellow phase between them.
* A **second pretrained model** detects emergency vehicles. When one is confirmed in any lane,
  **all other lanes turn RED instantly** and that lane stays GREEN until the emergency vehicle
  has gone (no detection for `emergency_clear_delay` seconds); the cycle then resumes with the next lane.
* **Admin panel**: upload/remove a video for each lane, tune settings, manage users, view the event log,
  and "Simulate emergency" button for demos.

## 1. Setup in VS Code

1. Install **Python 3.10 – 3.12**, open this folder in VS Code (`File → Open Folder`).
2. Open a terminal (`Ctrl + ~`) and run:

```bash
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

3. In VS Code press `Ctrl+Shift+P` → *Python: Select Interpreter* → choose the `venv` one.
4. Run:

```bash
python app.py
```

5. Open <http://127.0.0.1:5000> and sign in with **admin / admin123** (change the password in Admin).

`yolov8n.pt` downloads automatically on the first run (needs internet once).

## 2. Using it

1. Go to **Admin → Lane videos** and upload one video for each of the 4 lanes (mp4/avi/mov/mkv/webm).
   Videos loop. Use fixed-camera traffic clips for the best counts.
2. Open **Dashboard**: live annotated video, vehicle count, signal lamp and countdown for each lane,
   junction view, event log, and a red banner during an emergency.
3. Create extra `viewer` users in Admin if you want read-only logins.

## 3. Emergency-vehicle model (important)

The vehicle model (`yolov8n.pt`) is a general COCO model and does **not** know ambulances,
so the emergency detector is a second YOLO model that you place at:

```
models/emergency.pt
```

Get a YOLOv8 emergency-vehicle (ambulance / fire truck / police) `.pt` file from Roboflow Universe,
Hugging Face or GitHub (search "ambulance detection yolov8"), or train one in a few minutes:

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").train(data="path/to/emergency/data.yaml", epochs=50, imgsz=640)
# copy runs/detect/train/weights/best.pt  ->  models/emergency.pt
```

Class names containing `ambulance, fire, police, emergency, siren, rescue` count as emergency
(edit `EMERGENCY_KEYWORDS` in `config.py`). If the model's names contain none of them, every class is treated as emergency.

If `models/emergency.pt` is missing, the app falls back to a **demo-only heuristic** (red + blue
siren-light pixels on vehicles). It is not reliable; use it only until you add a real model.
The **Simulate emergency** button in Admin always works for demonstrating the preemption logic.

## 4. Project structure

```
app.py          Flask app: login, dashboard, admin, MJPEG streams, JSON API
controller.py   Signal logic (cyclic order, adaptive green, emergency preemption)
streams.py      Video readers per lane + YOLOv8 / emergency detection engine
db.py           SQLite: users, settings, event log
config.py       Paths, model names, lane names, default settings
templates/ static/   UI
uploads/        Uploaded lane videos          models/   emergency.pt goes here
```

## 5. Tuning & tips

* CPU is fine but slower; with an NVIDIA GPU install the CUDA build of PyTorch for faster detection.
* Counts are vehicles currently visible in the frame (queue density), re-read when the lane turns green.
* Change the secret key before deploying: `set SECRET_KEY=your-long-random-string`.
* Port busy? `set PORT=5001` then run again.

## 6. How the emergency logic works

1. Each lane's emergency flag turns on after 2 detections within the last 5 inferences (filters false hits).
2. The controller checks every 0.1 s. If a flag is on: lane → GREEN, others → RED immediately, mode = EMERGENCY.
3. If several lanes have emergencies, the one detected first is served first, then the next.
4. When the flag clears, that lane shows yellow and the normal cycle continues from the following lane.
