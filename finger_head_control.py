#!/usr/bin/env python3
"""
Combined controller:
  INDEX FINGER (calibrated) → cursor — absolute, like a precise touchpad
  OPEN MOUTH                → left click
  HEAD TILT LEFT            → scroll up
  HEAD TILT RIGHT           → scroll down
  HEAD UPRIGHT              → stop scroll
  c                         → recalibrate
  q                         → quit
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pynput.mouse import Controller as Mouse, Button
import time, os, subprocess

FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
HAND_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── Head-tilt scroll ──────────────────────────────────────────────────────────
LEFT_EYE  = 33;  RIGHT_EYE = 263
DEAD_ZONE = 8;   MAX_TILT  = 30;  SCROLL_CD = 0.10

# ── Mouth open → click ────────────────────────────────────────────────────────
MOUTH_TOP   = 13;  MOUTH_BOT  = 14
MOUTH_LEFT  = 78;  MOUTH_RIGHT= 308
MAR_THRESH  = 0.26
MOUTH_MIN_F = 3;   MOUTH_MAX_F = 30;  CLICK_CD = 0.80

# ── Finger cursor ─────────────────────────────────────────────────────────────
INDEX_TIP   = 8
FP_BUF_N    = 5      # median filter length on raw finger position (kills jitter)
CURSOR_SMOOTH = 0.40 # how fast cursor snaps to filtered target (higher = snappier)

# ── Calibration ───────────────────────────────────────────────────────────────
CALIB_PTS = [
    (c, r)
    for r in [0.15, 0.50, 0.85]
    for c in [0.15, 0.50, 0.85]
]  # 9-point 3×3 grid
CALIB_DWELL  = 2.0   # seconds to hold finger on each dot
CALIB_SETTLE = 0.5   # initial settling period to discard


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
    return (dx*dx + dy*dy) ** 0.5


def calc_mar(lm):
    v = lmd(lm, MOUTH_TOP, MOUTH_BOT)
    h = max(lmd(lm, MOUTH_LEFT, MOUTH_RIGHT), 1e-5)
    return v / h


def head_roll_deg(lm, w, h):
    lx, ly = lm[LEFT_EYE].x*w,  lm[LEFT_EYE].y*h
    rx, ry = lm[RIGHT_EYE].x*w, lm[RIGHT_EYE].y*h
    return float(np.degrees(np.arctan2(ry-ly, rx-lx)))


def poly_features(x, y):
    """Degree-2 polynomial features for mapping."""
    return np.array([1.0, x, y, x*x, y*y, x*y])


def fit_map(finger_pts, screen_norm_pts):
    A  = np.array([poly_features(p[0], p[1]) for p in finger_pts])
    bx = np.array([p[0] for p in screen_norm_pts])
    by = np.array([p[1] for p in screen_norm_pts])
    px, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    py, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    pred_x = A @ px;  pred_y = A @ py
    rmse = float(np.sqrt(np.mean((pred_x-bx)**2 + (pred_y-by)**2)))
    print(f"  Calibration RMSE: {rmse:.4f}  (lower = better)")
    return px, py


def apply_map(fx, fy, px, py, sw, sh):
    feat = poly_features(fx, fy)
    xn   = np.dot(feat, px)
    yn   = np.dot(feat, py)
    return (int(np.clip(xn*sw, 0, sw-1)),
            int(np.clip(yn*sh, 0, sh-1)))


def draw_tilt_gauge(frame, angle, fw, fh):
    cx, cy = fw//2, 55;  r = 42
    cv2.ellipse(frame,(cx,cy),(r,r),0,150,390,(50,50,50),3,cv2.LINE_AA)
    arc = 270 + np.clip(angle,-MAX_TILT,MAX_TILT)*(120/MAX_TILT)
    col = (0,200,200) if abs(angle)<DEAD_ZONE else ((0,220,0) if angle<0 else (0,80,255))
    nx = int(cx + r*np.cos(np.radians(arc)))
    ny = int(cy + r*np.sin(np.radians(arc)))
    cv2.line(frame,(cx,cy),(nx,ny),col,3,cv2.LINE_AA)
    cv2.circle(frame,(cx,cy),4,col,-1)
    cv2.putText(frame,f"{angle:+.1f}d",(cx-22,cy+r+14),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)


def hud(frame, text, color, fw):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(fw,30),(18,18,18),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    cv2.putText(frame,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.50,color,1,cv2.LINE_AA)


# ── Calibration UI ────────────────────────────────────────────────────────────

def calibrate(cap, hand_det, start_ms, sw, sh):
    cv2.namedWindow("calib", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    finger_data = []
    idx      = 0
    pt_start = None
    buf      = []

    print(f"Calibration: {len(CALIB_PTS)} points. Point your index finger at each dot.")

    while idx < len(CALIB_PTS):
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("calib")
            return None

        ts  = int(time.time()*1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = hand_det.detect_for_video(img, ts)
        now = time.time()

        finger_ok = False
        fx = fy = 0.0
        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            fx = lm[INDEX_TIP].x
            fy = lm[INDEX_TIP].y
            finger_ok = True

        if finger_ok:
            if pt_start is None:
                pt_start = now
            elapsed = now - pt_start
            if elapsed > CALIB_SETTLE:
                buf.append((fx, fy))
            if elapsed >= CALIB_DWELL + CALIB_SETTLE:
                med_x = float(np.median([p[0] for p in buf]))
                med_y = float(np.median([p[1] for p in buf]))
                finger_data.append((med_x, med_y))
                print(f"  [{idx+1}/{len(CALIB_PTS)}] screen=({CALIB_PTS[idx][0]:.2f},{CALIB_PTS[idx][1]:.2f})  finger=({med_x:.3f},{med_y:.3f})")
                idx     += 1
                pt_start = None
                buf      = []
        else:
            pt_start = None

        # ── Draw calibration screen ───────────────────────────────────────────
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        # Grid guide lines
        for r in [0.15,0.50,0.85]:
            y = int(r*sh); cv2.line(canvas,(0,y),(sw,y),(20,20,20),1)
        for c in [0.15,0.50,0.85]:
            x = int(c*sw); cv2.line(canvas,(x,0),(x,sh),(20,20,20),1)

        # Done dots
        for i in range(idx):
            ox,oy = int(CALIB_PTS[i][0]*sw), int(CALIB_PTS[i][1]*sh)
            cv2.circle(canvas,(ox,oy),8,(0,160,0),-1)
            cv2.putText(canvas,str(i+1),(ox-5,oy+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)

        # Future dots (faint)
        for i in range(idx+1, len(CALIB_PTS)):
            ox,oy = int(CALIB_PTS[i][0]*sw), int(CALIB_PTS[i][1]*sh)
            cv2.circle(canvas,(ox,oy),5,(35,35,35),-1)

        # Active dot
        if idx < len(CALIB_PTS):
            dx = int(CALIB_PTS[idx][0]*sw)
            dy = int(CALIB_PTS[idx][1]*sh)
            pulse = int(20 + 5*np.sin(now*7))
            cv2.circle(canvas,(dx,dy),pulse+8,(30,30,30),-1)
            cv2.circle(canvas,(dx,dy),pulse,(0,200,255),-1)
            cv2.circle(canvas,(dx,dy),5,(255,255,255),-1)
            if pt_start and finger_ok:
                elapsed  = now - pt_start
                progress = max(0.0, elapsed - CALIB_SETTLE) / CALIB_DWELL
                arc = int(min(progress,1.0)*360)
                cv2.ellipse(canvas,(dx,dy),(pulse+14,pulse+14),-90,0,arc,(0,255,120),3,cv2.LINE_AA)
            cv2.putText(canvas,str(idx+1),(dx+pulse+16,dy+5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),1)

        # Progress bar
        filled = int(sw * idx / len(CALIB_PTS))
        cv2.rectangle(canvas,(0,0),(sw,6),(30,30,30),-1)
        cv2.rectangle(canvas,(0,0),(filled,6),(0,200,120),-1)
        cv2.putText(canvas,f"Point {idx+1} / {len(CALIB_PTS)}",(sw//2-60,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(90,90,90),1)

        if not finger_ok:
            hint = "Raise your index finger and point at the dot"
            hcol = (0,140,255)
        else:
            remain = max(0.0,(CALIB_DWELL+CALIB_SETTLE)-(now-(pt_start or now)))
            hint   = f"Hold your finger on the dot  ({remain:.1f}s)"
            hcol   = (160,160,160)
        cv2.putText(canvas,hint,(sw//2-230,sh-50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,hcol,1,cv2.LINE_AA)

        # Camera preview (bottom-right)
        preview = cv2.resize(frame,(fw//4, fh//4))
        ph,pw = preview.shape[:2]
        canvas[sh-ph-12:sh-12, sw-pw-12:sw-12] = preview

        cv2.imshow("calib", canvas)

    cv2.destroyWindow("calib")

    if len(finger_data) < 6:
        print("Not enough calibration data."); return None

    px, py = fit_map(finger_data, CALIB_PTS)
    return px, py


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sw, sh = get_screen_size()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: camera not found"); return

    mouse = Mouse()

    face_opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    hand_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    face_det = mp.tasks.vision.FaceLandmarker.create_from_options(face_opts)
    hand_det = mp.tasks.vision.HandLandmarker.create_from_options(hand_opts)
    start_ms = int(time.time() * 1000)

    # Run calibration
    result = calibrate(cap, hand_det, start_ms, sw, sh)
    if result is None:
        cap.release(); face_det.close(); hand_det.close(); return
    px, py = result

    # Tracking state
    cx = float(sw // 2)
    cy = float(sh // 2)
    fp_buf_x = deque(maxlen=FP_BUF_N)   # median filter on raw finger position
    fp_buf_y = deque(maxlen=FP_BUF_N)

    mouth_frames = 0
    mouth_open   = False
    last_click   = 0.0

    angle_buf   = []
    last_scroll = 0.0

    cv2.namedWindow("Finger + Head Control")
    cv2.createTrackbar("Scroll speed", "Finger + Head Control", 3, 10, lambda _: None)

    print("Tracking.  Point finger → cursor | Open mouth → click | Tilt head → scroll | c=recalibrate | q=quit")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            result = calibrate(cap, hand_det, start_ms, sw, sh)
            if result:
                px, py = result

        scroll_sens = max(1, cv2.getTrackbarPos("Scroll speed", "Finger + Head Control"))

        ts  = int(time.time()*1000) - start_ms
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_res = face_det.detect_for_video(img, ts)
        hand_res = hand_det.detect_for_video(img, ts)
        now = time.time()

        stat_parts = []
        stat_col   = (0,180,80)

        # ── HAND: calibrated absolute cursor ─────────────────────────────────
        if hand_res.hand_landmarks:
            lm = hand_res.hand_landmarks[0]
            fx = lm[INDEX_TIP].x
            fy = lm[INDEX_TIP].y

            # Median-filter the raw finger position to kill camera jitter
            fp_buf_x.append(fx);  fp_buf_y.append(fy)
            fx_clean = float(np.median(fp_buf_x))
            fy_clean = float(np.median(fp_buf_y))

            # Map clean finger position to screen (calibrated)
            tx, ty = apply_map(fx_clean, fy_clean, px, py, sw, sh)

            # Responsive cursor smoothing — snaps quickly so small targets are reachable
            cx = CURSOR_SMOOTH * tx + (1 - CURSOR_SMOOTH) * cx
            cy = CURSOR_SMOOTH * ty + (1 - CURSOR_SMOOTH) * cy
            mouse.position = (int(cx), int(cy))

            # Draw fingertip
            ix = int(fx*fw); iy = int(fy*fh)
            cv2.circle(frame,(ix,iy),12,(0,220,255),-1)
            cv2.circle(frame,(ix,iy),12,(255,255,255),2)
            stat_parts.append(f"finger({fx:.2f},{fy:.2f})")
        else:
            fp_buf_x.clear();  fp_buf_y.clear()
            stat_parts.append("no hand — raise finger")

        # ── FACE: mouth click + head scroll ──────────────────────────────────
        if face_res.face_landmarks:
            lm = face_res.face_landmarks[0]

            # Mouth → click
            mar = calc_mar(lm)
            if mar > MAR_THRESH:
                mouth_frames += 1
                mouth_open    = True
            else:
                if mouth_open and MOUTH_MIN_F <= mouth_frames <= MOUTH_MAX_F:
                    if now - last_click >= CLICK_CD:
                        mouse.click(Button.left, 1)
                        last_click = now
                        stat_parts.append("CLICK!")
                mouth_frames = 0
                mouth_open   = False

            # MAR bar
            bar_h = int(np.clip(mar/0.80,0,1)*60)
            cv2.rectangle(frame,(5,fh-70),(14,fh-10),(40,40,40),-1)
            mar_col = (0,220,120) if mar > MAR_THRESH else (80,80,80)
            cv2.rectangle(frame,(5,fh-10-bar_h),(14,fh-10),mar_col,-1)
            cv2.putText(frame,"M",(2,fh-73),cv2.FONT_HERSHEY_SIMPLEX,0.3,(100,100,100),1)

            # Head tilt → scroll
            angle = head_roll_deg(lm, fw, fh)
            angle_buf.append(angle)
            if len(angle_buf) > 5: angle_buf.pop(0)
            angle = sum(angle_buf) / len(angle_buf)

            if angle < -DEAD_ZONE:
                direction = 1
                intensity = min((abs(angle)-DEAD_ZONE)/(MAX_TILT-DEAD_ZONE),1.0)
                eye_col   = (0,220,0)
                stat_parts.append(f"UP({angle:.1f}d)")
            elif angle > DEAD_ZONE:
                direction = -1
                intensity = min((angle-DEAD_ZONE)/(MAX_TILT-DEAD_ZONE),1.0)
                eye_col   = (0,80,255)
                stat_parts.append(f"DN({angle:.1f}d)")
            else:
                direction = 0; intensity = 0.0; eye_col = (0,200,200)
                stat_parts.append(f"still({angle:.1f}d)")

            if direction != 0 and now - last_scroll >= SCROLL_CD:
                mouse.scroll(0, direction * max(1, round(intensity*scroll_sens)))
                last_scroll = now

            lx=int(lm[LEFT_EYE].x*fw);  ly=int(lm[LEFT_EYE].y*fh)
            rx=int(lm[RIGHT_EYE].x*fw); ry=int(lm[RIGHT_EYE].y*fh)
            cv2.line(frame,(lx,ly),(rx,ry),eye_col,2,cv2.LINE_AA)
            cv2.circle(frame,(lx,ly),5,eye_col,-1)
            cv2.circle(frame,(rx,ry),5,eye_col,-1)
            draw_tilt_gauge(frame,angle,fw,fh)

            if direction != 0:
                acx,acy = fw//2, fh//2+20
                col = (0,220,0) if direction>0 else (0,80,255)
                cv2.arrowedLine(frame,(acx,acy+direction*35),(acx,acy-direction*35),
                                col,4,tipLength=0.35,line_type=cv2.LINE_AA)
        else:
            stat_parts.append("no face")

        hud(frame,"  |  ".join(stat_parts),stat_col,fw)
        cv2.imshow("Finger + Head Control",frame)

    cap.release()
    face_det.close()
    hand_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
