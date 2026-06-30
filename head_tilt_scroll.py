#!/usr/bin/env python3
"""
Head tilt scroll controller — completely hands-free.

  Tilt LEFT   -> scroll up
  Tilt RIGHT  -> scroll down
  Upright     -> stop  (dead zone in the middle)

Tilt further = scroll faster.
Sensitivity slider built into the window.  q = quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller as Mouse
import time, os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# Face landmarks used to compute head roll angle
# Left eye outer corner and right eye outer corner give a clean horizontal reference
LEFT_EYE  = 33    # left eye outer corner
RIGHT_EYE = 263   # right eye outer corner

# Tilt thresholds in degrees
DEAD_ZONE   = 8    # ±degrees from neutral before scroll starts
MAX_TILT    = 30   # degrees at which scroll hits full speed

COOLDOWN    = 0.10  # seconds between scroll events


def head_roll_deg(lm, w, h):
    """
    Compute head roll (tilt) angle in degrees from horizontal.
    Positive = tilted right (clockwise), negative = tilted left.
    """
    lx = lm[LEFT_EYE].x  * w
    ly = lm[LEFT_EYE].y  * h
    rx = lm[RIGHT_EYE].x * w
    ry = lm[RIGHT_EYE].y * h
    dx = rx - lx
    dy = ry - ly
    return float(np.degrees(np.arctan2(dy, dx)))


def draw_tilt_gauge(frame, angle, w, h):
    """
    Arc gauge at the top-centre showing tilt direction and amount.
    """
    cx, cy = w // 2, 55
    radius = 45

    # Background arc
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 150, 390,
                (50, 50, 50), 3, cv2.LINE_AA)

    # Clamp angle for display
    clamped = max(-MAX_TILT, min(MAX_TILT, angle))
    # Map angle to arc: centre of arc = 270° (pointing down), left = < 270, right = > 270
    arc_angle = 270 + clamped * (120 / MAX_TILT)

    if abs(angle) < DEAD_ZONE:
        col = (0, 200, 200)   # teal = dead zone
    elif angle < 0:
        col = (0, 220, 0)     # green = scroll up
    else:
        col = (0, 80, 255)    # red = scroll down

    # Needle line from centre toward arc angle
    nx = int(cx + radius * np.cos(np.radians(arc_angle)))
    ny = int(cy + radius * np.sin(np.radians(arc_angle)))
    cv2.line(frame, (cx, cy), (nx, ny), col, 3, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 5, col, -1)

    # Dead-zone markers
    for side in (-1, 1):
        a = 270 + side * DEAD_ZONE * (120 / MAX_TILT)
        mx = int(cx + radius * np.cos(np.radians(a)))
        my = int(cy + radius * np.sin(np.radians(a)))
        cv2.circle(frame, (mx, my), 3, (100, 100, 100), -1)

    # Angle label
    cv2.putText(frame, f"{angle:+.1f}deg", (cx - 30, cy + radius + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


def draw_eye_line(frame, lm, w, h, col):
    """Draw the eye-to-eye line used to measure tilt."""
    lx = int(lm[LEFT_EYE].x  * w)
    ly = int(lm[LEFT_EYE].y  * h)
    rx = int(lm[RIGHT_EYE].x * w)
    ry = int(lm[RIGHT_EYE].y * h)
    cv2.line(frame,  (lx, ly), (rx, ry), col, 2, cv2.LINE_AA)
    cv2.circle(frame, (lx, ly), 5, col, -1)
    cv2.circle(frame, (rx, ry), 5, col, -1)


def hud(frame, text, color, w):
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 32), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); return

    mouse   = Mouse()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    start_ms = int(time.time() * 1000)

    cv2.namedWindow("Head Tilt Scroll")
    cv2.createTrackbar("Sensitivity", "Head Tilt Scroll", 3, 10, lambda _: None)

    last_t    = 0.0
    angle_buf = []   # short smoothing buffer

    print("Head Tilt Scroll ready.  Tilt left=up  Tilt right=down  Upright=stop  q=quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sens = max(1, cv2.getTrackbarPos("Sensitivity", "Head Tilt Scroll"))

        ts  = int(time.time() * 1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(img, ts)

        stat_text  = "No face  |  q=quit"
        stat_color = (110, 110, 110)
        direction  = 0
        angle      = 0.0

        if res.face_landmarks:
            lm = res.face_landmarks[0]

            angle = head_roll_deg(lm, w, h)

            # Smooth over last 4 frames to kill jitter
            angle_buf.append(angle)
            if len(angle_buf) > 4:
                angle_buf.pop(0)
            angle = sum(angle_buf) / len(angle_buf)

            # ── Zone decision ─────────────────────────────────────────────────
            if angle < -DEAD_ZONE:
                # Tilted LEFT → scroll up
                direction = 1
                intensity = min((abs(angle) - DEAD_ZONE) / (MAX_TILT - DEAD_ZONE), 1.0)
                eye_col   = (0, 220, 0)
            elif angle > DEAD_ZONE:
                # Tilted RIGHT → scroll down
                direction = -1
                intensity = min((angle - DEAD_ZONE) / (MAX_TILT - DEAD_ZONE), 1.0)
                eye_col   = (0, 80, 255)
            else:
                direction = 0
                intensity = 0.0
                eye_col   = (0, 200, 200)

            draw_eye_line(frame, lm, w, h, eye_col)
            draw_tilt_gauge(frame, angle, w, h)

            # ── Emit scroll ───────────────────────────────────────────────────
            now = time.time()
            if direction != 0 and now - last_t >= COOLDOWN:
                amt = max(1, round(intensity * sens))
                mouse.scroll(0, direction * amt)
                last_t = now

            # ── Status ────────────────────────────────────────────────────────
            if direction == 1:
                stat_text, stat_color = f"SCROLL UP  (tilt {angle:.1f}deg)",   (0, 220, 0)
            elif direction == -1:
                stat_text, stat_color = f"SCROLL DOWN  (tilt {angle:.1f}deg)", (0, 80, 255)
            else:
                stat_text, stat_color = f"UPRIGHT - tilt to scroll  ({angle:.1f}deg)", (0, 200, 200)

        # ── Centre arrow ──────────────────────────────────────────────────────
        if direction != 0:
            cx, cy = w // 2, h // 2 + 30
            col    = (0, 220, 0) if direction > 0 else (0, 80, 255)
            cv2.arrowedLine(frame,
                            (cx, cy + direction * 40),
                            (cx, cy - direction * 40),
                            col, 4, tipLength=0.35, line_type=cv2.LINE_AA)

        hud(frame, stat_text, stat_color, w)
        cv2.imshow("Head Tilt Scroll", frame)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
