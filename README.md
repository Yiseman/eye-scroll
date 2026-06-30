# Eye Scroll — Webcam-Based Hands-Free Computer Control

Control your computer **without touching a mouse or keyboard** — using just your webcam. This project provides multiple input modes built on [MediaPipe](https://developers.google.com/mediapipe) and OpenCV.

---

## Modes

### `finger_head_control.py` — Main App (Recommended)
| Input | Action |
|---|---|
| Index finger position | Moves the cursor (calibrated, absolute) |
| Open mouth (small) | Left click |
| Tilt head left | Scroll up |
| Tilt head right | Scroll down |
| Press `c` | Recalibrate finger cursor |
| Press `q` | Quit |

### `eye_cursor.py` — Eye Gaze Cursor
| Input | Action |
|---|---|
| Eye gaze direction | Moves the cursor (16-point calibrated) |
| Double blink | Left click |
| Triple blink | Toggle cursor on/off |
| Press `c` | Recalibrate |
| Press `q` | Quit |

### `head_tilt_scroll.py` — Head Tilt Scrolling Only
| Input | Action |
|---|---|
| Tilt head left | Scroll up |
| Tilt head right | Scroll down |

### `gesture_control.py` — Two-Finger Pinch Scroll
| Input | Action |
|---|---|
| Pinch fingers together | Scroll down |
| Open pinch | Scroll up |

### `eye_scroll.py` — Basic Eye Scroll
Early prototype using eye position for scrolling.

---

## Requirements

- Python 3.9+
- Webcam
- Linux with X11 display (tested on Kali Linux)
- MediaPipe model files (see [Download Models](#download-models))

### Python packages
```
opencv-python>=4.8.0
mediapipe>=0.10.0
pynput>=1.7.6
numpy>=1.24.0
```

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/Yiseman/eye-scroll.git
cd eye-scroll
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models

The MediaPipe model files are not included in the repo (they are large binary files). Download them into the project root:

**Face Landmarker** (~3.7 MB):
```bash
curl -L -o face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

**Hand Landmarker** (~7.5 MB) — required for `finger_head_control.py`:
```bash
curl -L -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

---

## Usage

### Main app (finger + head + mouth)
```bash
DISPLAY=:0.0 venv/bin/python3 finger_head_control.py
```

### Eye gaze cursor
```bash
DISPLAY=:0.0 venv/bin/python3 eye_cursor.py
```

### Head tilt scroll only
```bash
DISPLAY=:0.0 venv/bin/python3 head_tilt_scroll.py
```

### Pinch gesture scroll
```bash
DISPLAY=:0.0 venv/bin/python3 gesture_control.py
```

> **Note:** `DISPLAY=:0.0` is required on Linux/Kali to route the OpenCV window to the X display.

---

## Calibration (finger_head_control.py and eye_cursor.py)

Both the finger cursor and eye cursor use a **polynomial regression calibration** for accurate screen mapping.

1. A fullscreen grid of dots appears on launch
2. Point your index finger (or look) at each glowing dot and hold for ~2 seconds
3. A progress ring fills around the dot — when complete it turns green and moves to the next
4. After all dots are collected the calibration is fitted and tracking begins
5. Press `c` at any time during tracking to recalibrate

The calibration corrects for camera angle, distance, and non-linear distortions, making the cursor precise even for small movements.

---

## How It Works

### Finger Cursor (`finger_head_control.py`)
- Detects index fingertip using MediaPipe Hand Landmarker in VIDEO mode
- Raw fingertip position is **median-filtered over 5 frames** to eliminate camera jitter
- A **degree-2 polynomial** maps filtered finger coordinates to screen coordinates using 9 calibration points
- Cursor position is smoothed with exponential smoothing (`CURSOR_SMOOTH = 0.40`) for fluid movement

### Eye Cursor (`eye_cursor.py`)
- Tracks iris position using MediaPipe Face Landmarker (iris landmarks 468, 473)
- **7-frame median filter** on raw gaze
- **16-point 4×4 calibration grid** with degree-2 polynomial regression
- Blink detection via Eye Aspect Ratio (EAR) — double blink = click, triple blink = toggle

### Head Tilt Scroll
- Computes **head roll angle** from left/right eye corner positions
- Dead zone (±8°) prevents accidental scroll from small movements
- Scroll speed scales with tilt angle up to MAX_TILT (30°)

### Mouth Click
- **Mouth Aspect Ratio (MAR)** = vertical opening / horizontal width
- Threshold `0.26` — triggers on a natural small mouth open
- Min 3 frames open required to prevent accidental triggers

---

## Troubleshooting

**Camera not found**
```bash
ls /dev/video*   # check camera index, change VideoCapture(0) if needed
```

**Cursor not moving / display error**
```bash
export DISPLAY=:0.0
# or run with: DISPLAY=:0.0 python3 ...
```

**Camera locked by another app** (e.g. browser)
Close the other app using the camera, then relaunch.

**Too sensitive / shaking**
The 5-frame median filter handles most jitter. If still shaky, increase `FP_BUF_N` in `finger_head_control.py`.

**Mouth click not triggering**
Lower `MAR_THRESH` in `finger_head_control.py` (default `0.26`). If triggering too easily, raise it.

---

## Tech Stack

- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/overview) — face and hand landmark detection
- [OpenCV](https://opencv.org/) — webcam capture and UI
- [pynput](https://pynput.readthedocs.io/) — mouse control
- [NumPy](https://numpy.org/) — polynomial regression, median filtering

---

## License

MIT
