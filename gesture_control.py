#!/usr/bin/env python3
"""
Two-finger scroll controller.

  Pinch closed  (thumb + index together) -> scroll DOWN continuously
  Wide open     (thumb + index spread)   -> scroll UP continuously
  Middle range  (small gap)              -> STOP

Sensitivity slider is built into the window.  q = quit.
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller as Mouse
import time, os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

WRIST     = 0
MID_MCP   = 9    # middle-finger knuckle — used as hand-size reference
THUMB_TIP = 4
INDEX_TIP = 8

# Thresholds in units of (pinch_dist / hand_size)
CLOSE_T   = 0.13   # below → scroll down (nearly touching only)
OPEN_T    = 0.75   # above → scroll up
# gap between CLOSE_T and OPEN_T is the dead zone (stop)

COOLDOWN  = 0.12   # seconds between scroll events (controls max rate)


def ndist(lm, a, b):
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return (dx * dx + dy * dy) ** 0.5


def draw_gauge(frame, ratio, x, y, height=120):
    """
    Vertical bar on the side:
      bottom = fully closed (down)
      top    = fully open   (up)
      middle band = dead zone
    """
    w = 18
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + height), (40, 40, 40), -1)

    # Dead-zone band
    dz_top = int(y + (1 - OPEN_T / 1.2)  * height)
    dz_bot = int(y + (1 - CLOSE_T / 1.2) * height)
    cv2.rectangle(frame, (x, dz_top), (x + w, dz_bot), (60, 60, 0), -1)

    # Current-position dot
    clamped = min(ratio / 1.2, 1.0)
    dot_y   = int(y + height - clamped * height)
    if ratio < CLOSE_T:
        col = (0, 60, 255)
    elif ratio > OPEN_T:
        col = (0, 220, 0)
    else:
        col = (0, 200, 200)
    cv2.circle(frame, (x + w // 2, dot_y), 8, col, -1)
    cv2.circle(frame, (x + w // 2, dot_y), 8, (255, 255, 255), 1)

    # Labels
    cv2.putText(frame, "UP",   (x, y - 4),          cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0),   1)
    cv2.putText(frame, "DN",   (x, y + height + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 60, 255),  1)
    cv2.putText(frame, "STOP", (x - 2, dz_bot - 4),  cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 200), 1)


def hud(frame, text, color):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 32), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); return

    mouse   = Mouse()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    start_ms = int(time.time() * 1000)

    # Create window + sensitivity slider
    cv2.namedWindow("Scroll Control")
    cv2.createTrackbar("Sensitivity", "Scroll Control", 2, 20, lambda _: None)

    last_t = 0.0

    print("Two-finger Scroll ready.  Pinch=down  Open=up  Dead-zone=stop  q=quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sens = max(1, cv2.getTrackbarPos("Sensitivity", "Scroll Control"))

        ts  = int(time.time() * 1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(img, ts)

        stat_text  = "No hand  |  q=quit"
        stat_color = (110, 110, 110)
        direction  = 0
        ratio      = 0.5   # default mid (for gauge)

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]

            hand_size = max(ndist(lm, WRIST, MID_MCP), 1e-4)
            ratio     = ndist(lm, THUMB_TIP, INDEX_TIP) / hand_size

            # ── Finger dot positions ──────────────────────────────────────────
            tx, ty = int(lm[THUMB_TIP].x * w), int(lm[THUMB_TIP].y * h)
            ix, iy = int(lm[INDEX_TIP].x * w), int(lm[INDEX_TIP].y * h)

            # ── Zone decision ─────────────────────────────────────────────────
            if ratio < CLOSE_T:
                direction = -1               # scroll down
                intensity = 1.0 - (ratio / CLOSE_T)   # 0..1, higher = more closed
                line_col  = (0, 60, 255)
            elif ratio > OPEN_T:
                direction = 1                # scroll up
                intensity = min((ratio - OPEN_T) / 0.4, 1.0)  # 0..1, higher = more open
                line_col  = (0, 220, 0)
            else:
                direction = 0                # dead zone
                intensity = 0.0
                line_col  = (0, 200, 200)

            # ── Draw connecting line + fingertip dots ─────────────────────────
            cv2.line(frame, (tx, ty), (ix, iy), line_col, 3, cv2.LINE_AA)

            cv2.circle(frame, (tx, ty), 12, (255, 140, 0), -1)   # thumb - orange
            cv2.circle(frame, (tx, ty), 12, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.circle(frame, (ix, iy), 12, (0, 200, 255), -1)   # index - cyan
            cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2, cv2.LINE_AA)

            # ── Emit scroll ───────────────────────────────────────────────────
            now = time.time()
            if direction != 0 and now - last_t >= COOLDOWN:
                amt = max(1, round(intensity * sens))
                mouse.scroll(0, direction * amt)
                last_t = now

            # ── Status text ───────────────────────────────────────────────────
            if direction == 1:
                stat_text, stat_color = f"SCROLL UP  (open {ratio:.2f})",   (0, 220, 0)
            elif direction == -1:
                stat_text, stat_color = f"SCROLL DOWN  (pinch {ratio:.2f})", (0, 80, 255)
            else:
                stat_text, stat_color = f"STOP  (open wider or pinch)  {ratio:.2f}", (0, 200, 200)

        # ── Gauge bar (right side) ────────────────────────────────────────────
        draw_gauge(frame, ratio, w - 30, 40)

        # ── Arrow indicator ───────────────────────────────────────────────────
        if direction != 0:
            cx, cy = w // 2, h // 2
            col    = (0, 220, 0) if direction > 0 else (0, 80, 255)
            cv2.arrowedLine(frame,
                            (cx, cy + direction * 45),
                            (cx, cy - direction * 45),
                            col, 4, tipLength=0.35, line_type=cv2.LINE_AA)

        hud(frame, stat_text, stat_color)
        cv2.imshow("Scroll Control", frame)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
