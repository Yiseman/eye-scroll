#!/usr/bin/env python3
"""
Eye-tracking scroll controller.
  Look UP   → scroll up
  Look DOWN → scroll down
  Center    → stop

Controls:
  q  — quit
  p  — pause / resume
  c  — recalibrate (stare straight ahead for 2 seconds)

First run downloads the face landmarker model (~30 MB).
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller as Mouse
import time
import urllib.request
import os
import sys

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── Iris / eye landmark indices (same for old and new API) ───────────────────
L_IRIS, R_IRIS    = 468, 473   # iris centres (requires the full 478-landmark model)
L_EYE_T, L_EYE_B = 159, 145   # left eye top / bottom lid
R_EYE_T, R_EYE_B = 386, 374   # right eye top / bottom lid

# ── Tuning ───────────────────────────────────────────────────────────────────
SCROLL_THRESHOLD = 0.12   # deviation from neutral before scroll triggers
SCROLL_MAX_SPEED = 5      # max scroll units per tick
SCROLL_INTERVAL  = 0.08   # seconds between scroll events
CALIB_SECONDS    = 2.0    # look straight ahead this long during calibration


# ── Helpers ──────────────────────────────────────────────────────────────────

def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading face landmarker model (~30 MB) — one-time setup…")

    def progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:.0f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress)
    print("\nDownload complete.")


def gaze_ratio(lm, iris, top, bot, h):
    """Vertical iris position within the eye opening (0 = top, 1 = bottom)."""
    iy = lm[iris].y * h
    ty = lm[top].y  * h
    by = lm[bot].y  * h
    span = by - ty
    return (iy - ty) / span if span > 1 else 0.5


def scroll_amount(deviation, threshold, max_speed):
    """Map gaze deviation → scroll units (positive = up, negative = down)."""
    if abs(deviation) < threshold:
        return 0
    t = min((abs(deviation) - threshold) / max(0.5 - threshold, 0.01), 1.0)
    units = max(1, int(t * max_speed))
    return units if deviation < 0 else -units   # negative dev = looking up


def draw_gaze_bar(frame, ratio, x, y, label):
    bh, bw = 60, 10
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (40, 40, 40), -1)
    dot_y = int(np.clip(y + ratio * bh, y, y + bh))
    cv2.circle(frame, (x + bw // 2, dot_y), 4, (0, 220, 100), -1)
    cv2.putText(frame, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)


def draw_arrow(frame, direction, cx, cy):
    if direction == 0:
        return
    length = 30
    dy = -direction             # direction > 0 → arrow points up on screen
    pt1 = (cx, cy - dy * length)
    pt2 = (cx, cy + dy * length)
    color = (0, 220, 0) if direction > 0 else (0, 80, 255)
    cv2.arrowedLine(frame, pt2, pt1, color, 3, tipLength=0.3)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera. Try changing VideoCapture(0) to VideoCapture(1).")
        return

    mouse       = Mouse()
    paused      = False
    last_scroll = 0.0
    neutral     = 0.5
    calib_buf   = []
    calibrating = False
    calib_start = 0.0

    # ── Build face landmarker (Tasks API, VIDEO mode for tracking) ────────────
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    print("Eye Scroll running — look straight ahead, then press c to calibrate.")
    print("  q = quit   p = pause/resume   c = calibrate")

    start_ms = int(time.time() * 1000)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        cx, cy = w // 2, h // 2

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('c'):
            calibrating = True
            calib_buf   = []
            calib_start = time.time()
            print("Calibrating — stare straight ahead…")

        # Monotonically increasing timestamp required by VIDEO mode
        timestamp_ms = int(time.time() * 1000) - start_ms

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, timestamp_ms)

        status_text  = "No face detected"
        status_color = (100, 100, 100)
        scroll_dir   = 0

        if result.face_landmarks:
            lm  = result.face_landmarks[0]
            lr  = gaze_ratio(lm, L_IRIS, L_EYE_T, L_EYE_B, h)
            rr  = gaze_ratio(lm, R_IRIS, R_EYE_T, R_EYE_B, h)
            avg = (lr + rr) / 2

            # ── Calibration ──────────────────────────────────────────────────
            if calibrating:
                calib_buf.append(avg)
                elapsed = time.time() - calib_start
                remain  = max(0.0, CALIB_SECONDS - elapsed)
                cv2.putText(frame, f"Calibrating… {remain:.1f}s",
                            (cx - 110, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                if elapsed >= CALIB_SECONDS:
                    neutral = float(np.median(calib_buf))
                    calibrating = False
                    print(f"Calibration done — neutral gaze = {neutral:.3f}")
            else:
                dev = avg - neutral   # negative = looking up, positive = down

                if not paused:
                    now = time.time()
                    if now - last_scroll >= SCROLL_INTERVAL:
                        amt = scroll_amount(dev, SCROLL_THRESHOLD, SCROLL_MAX_SPEED)
                        if amt:
                            mouse.scroll(0, amt)
                            scroll_dir = 1 if amt > 0 else -1
                        last_scroll = now

                if paused:
                    status_text, status_color = "PAUSED  (p to resume)", (200, 180, 0)
                elif dev < -SCROLL_THRESHOLD:
                    status_text, status_color = f"SCROLL UP   (dev {dev:.2f})", (0, 220, 0)
                elif dev > SCROLL_THRESHOLD:
                    status_text, status_color = f"SCROLL DOWN (dev {dev:.2f})", (0, 80, 255)
                else:
                    status_text, status_color = "CENTER", (200, 200, 0)

            # ── Draw iris dots ────────────────────────────────────────────────
            for idx in (L_IRIS, R_IRIS):
                px = int(lm[idx].x * w)
                py = int(lm[idx].y * h)
                cv2.circle(frame, (px, py), 5, (0, 200, 255), -1)

            # ── Gaze bars (bottom-left corner) ────────────────────────────────
            draw_gaze_bar(frame, lr, 10, h - 80, "L")
            draw_gaze_bar(frame, rr, 30, h - 80, "R")

        # ── Scroll arrow in frame centre ──────────────────────────────────────
        draw_arrow(frame, scroll_dir, cx, cy)

        # ── Status bar ────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 28), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame,
                    f"{status_text}   |   q=quit  p=pause  c=calibrate",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        cv2.imshow("Eye Scroll", frame)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
