#!/usr/bin/env python3
"""
Eye-controlled cursor — high-accuracy edition.

  Eyes move        → cursor follows gaze
  Double blink     → left click  (fires 0.4s after 2nd blink)
  Triple blink     → toggle cursor ON / OFF
  c                → redo calibration
  q                → quit

Calibration: 16-point 4×4 grid + degree-2 polynomial regression.
"""

import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller as Mouse, Button
import time, os, subprocess

FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ── Landmark indices ──────────────────────────────────────────────────────────
L_IRIS = 468;  R_IRIS = 473

# Left eye — 3 vertical pairs + horizontal
L_EAR_PAIRS = [(159, 145), (158, 153), (157, 154)]
L_OUTER = 33;  L_INNER = 133
L_TOP   = 159; L_BOT   = 145

# Right eye
R_EAR_PAIRS = [(386, 374), (385, 380), (384, 381)]
R_OUTER = 362; R_INNER = 263
R_TOP   = 386; R_BOT   = 374

# ── Calibration ───────────────────────────────────────────────────────────────
# 16-point 4×4 grid
CALIB_PTS = [
    (c, r)
    for r in [0.08, 0.33, 0.67, 0.92]
    for c in [0.08, 0.33, 0.67, 0.92]
]
CALIB_DWELL  = 2.5   # seconds holding gaze per dot
CALIB_SETTLE = 0.5   # initial settling period to discard

# ── Blink detection ───────────────────────────────────────────────────────────
EAR_THRESH       = 0.18
BLINK_MIN_F      = 2     # min frames eye must be closed
BLINK_MAX_F      = 16    # max frames before it's a deliberate hold, not a blink
DOUBLE_WIN       = 0.60  # seconds — 2 blinks in this window = click
TRIPLE_WIN       = 1.20  # seconds — 3 blinks in this window = toggle
CLICK_DELAY      = 0.40  # wait this long after 2nd blink before firing (in case 3rd comes)
CLICK_COOLDOWN   = 0.80

# ── Cursor ────────────────────────────────────────────────────────────────────
SMOOTH_BASE  = 0.07   # exponential smoothing (lower = smoother, more lag)
GAZE_HIST_N  = 7      # median filter length on raw gaze


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_screen_size():
    try:
        out = subprocess.check_output(
            ["xdpyinfo"], env={**os.environ, "DISPLAY": ":0.0"}
        ).decode()
        for line in out.splitlines():
            if "dimensions:" in line:
                w, h = map(int, line.strip().split()[1].split("x"))
                return w, h
    except Exception:
        pass
    return 1920, 1080


def lmd(lm, a, b):
    dx, dy = lm[a].x - lm[b].x, lm[a].y - lm[b].y
    return (dx * dx + dy * dy) ** 0.5


def calc_ear(lm):
    """EAR using 3 vertical pairs per eye, averaged across both eyes."""
    def eye_ear(pairs, outer, inner):
        v = sum(lmd(lm, t, b) for t, b in pairs) / len(pairs)
        h = max(lmd(lm, outer, inner), 1e-5)
        return v / h
    return (eye_ear(L_EAR_PAIRS, L_OUTER, L_INNER) +
            eye_ear(R_EAR_PAIRS, R_OUTER, R_INNER)) / 2


def calc_gaze(lm):
    """
    Raw gaze ratios (h, v) averaged across both irises.
    h: 0=left  1=right
    v: 0=up    1=down
    """
    def ratio(iris, lo, li, lt, lb):
        h = (lm[iris].x - lm[lo].x) / max(lm[li].x - lm[lo].x, 1e-5)
        v = (lm[iris].y - lm[lt].y) / max(lm[lb].y - lm[lt].y, 1e-5)
        return h, v
    lh, lv = ratio(L_IRIS, L_OUTER, L_INNER, L_TOP, L_BOT)
    rh, rv  = ratio(R_IRIS, R_OUTER, R_INNER, R_TOP, R_BOT)
    return (lh + rh) / 2, (lv + rv) / 2


def poly_features(h, v):
    """Degree-2 polynomial features: [1, h, v, h², v², h·v]."""
    return np.array([1.0, h, v, h * h, v * v, h * v])


def fit_poly_map(gaze_pts, screen_norm_pts):
    """Fit degree-2 polynomial: gaze (h,v) → normalised screen (x, y)."""
    A  = np.array([poly_features(g[0], g[1]) for g in gaze_pts])
    bx = np.array([p[0] for p in screen_norm_pts])
    by = np.array([p[1] for p in screen_norm_pts])
    px, res_x, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    py, res_y, _, _ = np.linalg.lstsq(A, by, rcond=None)

    # Report residual accuracy
    pred_x = A @ px;  pred_y = A @ py
    rmse_x = float(np.sqrt(np.mean((pred_x - bx) ** 2)))
    rmse_y = float(np.sqrt(np.mean((pred_y - by) ** 2)))
    print(f"  Calibration fit RMSE: x={rmse_x:.4f}  y={rmse_y:.4f}  (lower=better)")
    return px, py


def apply_map(gh, gv, px, py, sw, sh):
    feat = poly_features(gh, gv)
    xn   = np.dot(feat, px)
    yn   = np.dot(feat, py)
    return (int(np.clip(xn * sw, 0, sw - 1)),
            int(np.clip(yn * sh, 0, sh - 1)))


# ── Calibration UI ────────────────────────────────────────────────────────────

def calibrate(cap, detector, start_ms, sw, sh):
    cv2.namedWindow("calib", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    gaze_data = []
    idx       = 0
    pt_start  = None
    buf       = []

    print(f"Calibration: {len(CALIB_PTS)} points, {CALIB_DWELL}s each. Look at each dot.")

    while idx < len(CALIB_PTS):
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow("calib")
            return None

        ts  = int(time.time() * 1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(img, ts)
        now = time.time()

        face_ok  = bool(res.face_landmarks)
        eye_open = False
        gh = gv  = 0.0

        if face_ok:
            lm       = res.face_landmarks[0]
            gh, gv   = calc_gaze(lm)
            ear_val  = calc_ear(lm)
            eye_open = ear_val > EAR_THRESH

            if pt_start is None and eye_open:
                pt_start = now
            if pt_start and eye_open:
                elapsed = now - pt_start
                if elapsed > CALIB_SETTLE:
                    buf.append((gh, gv))
                if elapsed >= CALIB_DWELL + CALIB_SETTLE:
                    med_h = float(np.median([g[0] for g in buf]))
                    med_v = float(np.median([g[1] for g in buf]))
                    gaze_data.append((med_h, med_v))
                    print(f"  [{idx+1:02d}/{len(CALIB_PTS)}] screen=({CALIB_PTS[idx][0]:.2f},{CALIB_PTS[idx][1]:.2f})  gaze=({med_h:.3f},{med_v:.3f})")
                    idx     += 1
                    pt_start = None
                    buf      = []
            elif not eye_open:
                pt_start = None
        else:
            pt_start = None

        # ── Draw calibration frame ────────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        # Grid guide lines (very subtle)
        for r in [0.08, 0.33, 0.67, 0.92]:
            y = int(r * sh)
            cv2.line(canvas, (0, y), (sw, y), (20, 20, 20), 1)
        for c in [0.08, 0.33, 0.67, 0.92]:
            x = int(c * sw)
            cv2.line(canvas, (x, 0), (x, sh), (20, 20, 20), 1)

        # Completed dots
        for i in range(idx):
            ox = int(CALIB_PTS[i][0] * sw)
            oy = int(CALIB_PTS[i][1] * sh)
            cv2.circle(canvas, (ox, oy), 8, (0, 160, 0), -1)
            cv2.putText(canvas, str(i + 1), (ox - 5, oy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Future dots (faint)
        for i in range(idx + 1, len(CALIB_PTS)):
            ox = int(CALIB_PTS[i][0] * sw)
            oy = int(CALIB_PTS[i][1] * sh)
            cv2.circle(canvas, (ox, oy), 5, (35, 35, 35), -1)

        # Current active dot
        if idx < len(CALIB_PTS):
            dx = int(CALIB_PTS[idx][0] * sw)
            dy = int(CALIB_PTS[idx][1] * sh)
            pulse = int(20 + 5 * np.sin(now * 7))

            cv2.circle(canvas, (dx, dy), pulse + 8, (30, 30, 30), -1)
            cv2.circle(canvas, (dx, dy), pulse, (0, 200, 255), -1)
            cv2.circle(canvas, (dx, dy), 5, (255, 255, 255), -1)

            # Progress ring
            if pt_start and face_ok and eye_open:
                elapsed  = now - pt_start
                progress = max(0.0, elapsed - CALIB_SETTLE) / CALIB_DWELL
                arc = int(min(progress, 1.0) * 360)
                cv2.ellipse(canvas, (dx, dy), (pulse + 14, pulse + 14),
                            -90, 0, arc, (0, 255, 120), 3, cv2.LINE_AA)

            # Point number next to dot
            cv2.putText(canvas, str(idx + 1), (dx + pulse + 16, dy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        # Status / hint
        if not face_ok:
            hint, hcol = "Face not detected — move into camera frame", (0, 60, 255)
        elif not eye_open:
            hint, hcol = "Open your eyes wide and look at the dot", (0, 140, 255)
        else:
            remain = max(0.0, (CALIB_DWELL + CALIB_SETTLE) - (now - (pt_start or now)))
            hint   = f"Hold your gaze on the dot  ({remain:.1f}s)"
            hcol   = (160, 160, 160)

        cv2.putText(canvas, hint, (sw // 2 - 230, sh - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, hcol, 1, cv2.LINE_AA)

        # Progress bar at top
        filled = int(sw * idx / len(CALIB_PTS))
        cv2.rectangle(canvas, (0, 0), (sw, 6), (30, 30, 30), -1)
        cv2.rectangle(canvas, (0, 0), (filled, 6), (0, 200, 120), -1)
        cv2.putText(canvas, f"Point {idx + 1} / {len(CALIB_PTS)}",
                    (sw // 2 - 60, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 90, 90), 1)

        # Camera preview (bottom-right)
        preview   = cv2.resize(frame, (fw // 4, fh // 4))
        ph, pw    = preview.shape[:2]
        canvas[sh - ph - 12: sh - 12, sw - pw - 12: sw - 12] = preview

        cv2.imshow("calib", canvas)

    cv2.destroyWindow("calib")

    if len(gaze_data) < 6:
        print("Not enough calibration data."); return None

    px, py = fit_poly_map(gaze_data, CALIB_PTS)
    return px, py


# ── Main tracking loop ────────────────────────────────────────────────────────

def main():
    sw, sh = get_screen_size()
    print(f"Screen: {sw}x{sh}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); return

    mouse   = Mouse()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.65,
        min_face_presence_confidence=0.65,
        min_tracking_confidence=0.65,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    start_ms = int(time.time() * 1000)

    # Calibrate
    result = calibrate(cap, detector, start_ms, sw, sh)
    if result is None:
        cap.release(); detector.close(); return
    px, py = result

    # Cursor state
    cx = float(sw // 2)
    cy = float(sh // 2)
    gaze_h_buf = []   # raw gaze history for median filter
    gaze_v_buf = []

    # Blink state machine
    blink_frames  = 0
    eyes_closed   = False
    blink_times   = []        # timestamps of confirmed blinks
    pending_click = None      # time to fire a click (if no 3rd blink comes)
    last_click_t  = 0.0

    # Cursor toggle
    cursor_active = True

    cv2.namedWindow("Eye Cursor")
    print("Tracking active.  Double-blink=click  Triple-blink=pause/resume  c=recalibrate  q=quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            result = calibrate(cap, detector, start_ms, sw, sh)
            if result:
                px, py = result
                gaze_h_buf.clear()
                gaze_v_buf.clear()

        ts  = int(time.time() * 1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(img, ts)
        now = time.time()

        # ── Fire pending click if no 3rd blink arrived ────────────────────────
        if pending_click and now >= pending_click:
            if now - last_click_t >= CLICK_COOLDOWN:
                mouse.click(Button.left, 1)
                last_click_t = now
            pending_click = None

        stat_text  = "No face  |  c=recalibrate  q=quit"
        stat_color = (110, 110, 110)
        action_msg = ""

        if res.face_landmarks:
            lm      = res.face_landmarks[0]
            ear_val = calc_ear(lm)
            gh, gv  = calc_gaze(lm)

            # ── Gaze median filter ────────────────────────────────────────────
            gaze_h_buf.append(gh);  gaze_v_buf.append(gv)
            if len(gaze_h_buf) > GAZE_HIST_N:
                gaze_h_buf.pop(0);  gaze_v_buf.pop(0)
            smooth_gh = float(np.median(gaze_h_buf))
            smooth_gv = float(np.median(gaze_v_buf))

            # ── Cursor movement ───────────────────────────────────────────────
            if cursor_active and ear_val >= EAR_THRESH:
                tx, ty = apply_map(smooth_gh, smooth_gv, px, py, sw, sh)

                # Velocity-adaptive smoothing: move faster when gaze moves fast
                vel    = abs(tx - cx) + abs(ty - cy)
                alpha  = min(SMOOTH_BASE + vel * 0.0003, 0.35)
                cx     = alpha * tx + (1 - alpha) * cx
                cy     = alpha * ty + (1 - alpha) * cy
                mouse.position = (int(cx), int(cy))

            # ── Blink state machine ───────────────────────────────────────────
            if ear_val < EAR_THRESH:
                blink_frames += 1
                eyes_closed   = True
            else:
                if eyes_closed and BLINK_MIN_F <= blink_frames <= BLINK_MAX_F:
                    # Valid blink confirmed
                    blink_times.append(now)

                    # Expire old blinks outside triple-blink window
                    blink_times = [t for t in blink_times if now - t <= TRIPLE_WIN]

                    recent3 = [t for t in blink_times if now - t <= TRIPLE_WIN]
                    recent2 = [t for t in blink_times if now - t <= DOUBLE_WIN]

                    if len(recent3) >= 3:
                        # TRIPLE BLINK → toggle cursor
                        cursor_active = not cursor_active
                        blink_times   = []
                        pending_click = None
                        action_msg    = "CURSOR ON" if cursor_active else "CURSOR PAUSED"
                        print(f"  Triple blink → cursor {'ON' if cursor_active else 'OFF'}")

                    elif len(recent2) >= 2 and not pending_click:
                        # DOUBLE BLINK → schedule click
                        pending_click = now + CLICK_DELAY

                blink_frames = 0
                eyes_closed  = False

            # Expire stale blink history
            blink_times = [t for t in blink_times if now - t <= TRIPLE_WIN]

            # ── Draw iris dots ────────────────────────────────────────────────
            for iris in (L_IRIS, R_IRIS):
                ix = int(lm[iris].x * fw)
                iy = int(lm[iris].y * fh)
                col = (0, 60, 255) if ear_val < EAR_THRESH else (0, 200, 255)
                cv2.circle(frame, (ix, iy), 5, col, -1)
                cv2.circle(frame, (ix, iy), 5, (255, 255, 255), 1)

            # EAR bar (left edge)
            bar_h = int(np.clip(ear_val / 0.35, 0, 1) * 70)
            cv2.rectangle(frame, (5, fh - 80), (15, fh - 10), (40, 40, 40), -1)
            ear_col = (0, 60, 255) if ear_val < EAR_THRESH else (0, 220, 0)
            cv2.rectangle(frame, (5, fh - 10 - bar_h), (15, fh - 10), ear_col, -1)
            cv2.putText(frame, "EAR", (1, fh - 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100, 100, 100), 1)

            # Blink counter dots
            for i, _ in enumerate(blink_times[-3:]):
                cv2.circle(frame, (fw - 20 - i * 16, fh - 15), 6, (0, 180, 255), -1)

            # Status
            if action_msg:
                stat_text  = action_msg
                stat_color = (0, 220, 180)
            elif pending_click:
                stat_text  = "CLICK incoming..."
                stat_color = (50, 50, 255)
            elif not cursor_active:
                stat_text  = "CURSOR PAUSED  (triple-blink to resume)"
                stat_color = (0, 140, 255)
            else:
                bc         = len(blink_times)
                stat_text  = f"EAR {ear_val:.2f}  blinks:{bc}  |  dbl-blink=click  triple=pause  c=recal"
                stat_color = (0, 180, 80)

        # ── Cursor-paused overlay ─────────────────────────────────────────────
        if not cursor_active:
            cv2.rectangle(frame, (fw - 16, 33), (fw, fh), (0, 0, 80), -1)
            cv2.putText(frame, "OFF", (fw - 15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 100, 255), 1)

        # ── HUD ───────────────────────────────────────────────────────────────
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (fw, 30), (18, 18, 18), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, stat_text, (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, stat_color, 1, cv2.LINE_AA)

        cv2.imshow("Eye Cursor", frame)

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
