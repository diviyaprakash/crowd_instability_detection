"""
Smart Detection — Rule-based using Motion Entropy + Directional Variance
Bypasses the neural network and uses the core motion coherence metrics directly.
This is actually the core contribution of the project.
"""

import cv2
import numpy as np
from collections import deque

# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH     = r"D:\Minor_Project\Crowd Jumps at a Concert _ Copyright Free Footage.mp4"   # change this
SEQ_LEN        = 16
ENTROPY_THRESH = 0.50    # high entropy = chaotic/abnormal motion
VAR_THRESH     = 0.35    # high variance = random directions = abnormal
WINDOW         = 8       # smoothing window
CONFIRM        = 3       # consecutive windows before alert

# ── OPTICAL FLOW ──────────────────────────────────────────────────────────────
def compute_flow(prev, curr):
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow    = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx, dy  = flow[..., 0], flow[..., 1]
    mag, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    mag_r   = cv2.resize(mag,   (64, 64))
    angle_r = cv2.resize(angle, (64, 64))
    return mag_r, angle_r

# ── MOTION ENTROPY ────────────────────────────────────────────────────────────
def motion_entropy(mag_seq):
    """High entropy = chaotic magnitude distribution = abnormal"""
    frames = np.stack(mag_seq)
    flat   = frames.ravel()
    norm   = flat / (flat.max() + 1e-8)
    hist, _ = np.histogram(norm, bins=32, range=(0,1), density=True)
    hist    = hist + 1e-9
    hist    = hist / hist.sum()
    ent     = -np.sum(hist * np.log2(hist))
    return float(ent / np.log2(32))   # normalise to [0,1]

# ── DIRECTIONAL VARIANCE ──────────────────────────────────────────────────────
def directional_variance(angle_seq):
    """High variance = crowd moving in random directions = abnormal"""
    frames = np.stack(angle_seq)
    rad    = np.deg2rad(frames)
    R      = np.sqrt(np.mean(np.sin(rad))**2 + np.mean(np.cos(rad))**2)
    return float(1.0 - R)   # 0=coherent, 1=chaotic

# ── MAIN ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev = cap.read()

mag_seq    = []
angle_seq  = []
ent_hist   = deque(maxlen=WINDOW)
var_hist   = deque(maxlen=WINDOW)
cooldown   = 0
frame_idx  = 0
alert_count = 0

print(f"{'Frame':>6} | {'Entropy':>8} | {'DirVar':>8} | {'AvgEnt':>8} | {'AvgVar':>8} | Status")
print("-" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mag, angle = compute_flow(prev, frame)
    mag_seq.append(mag)
    angle_seq.append(angle)

    if len(mag_seq) == SEQ_LEN:
        ent = motion_entropy(mag_seq)
        var = directional_variance(angle_seq)

        ent_hist.append(ent)
        var_hist.append(var)

        avg_ent = float(np.mean(ent_hist))
        avg_var = float(np.mean(var_hist))

        # ── Decision ──────────────────────────────────────────────────────────
        high_ent = sum(e > ENTROPY_THRESH for e in ent_hist)
        high_var = sum(v > VAR_THRESH     for v in var_hist)

        is_abnormal   = high_ent >= CONFIRM and high_var >= CONFIRM
        early_warning = (avg_ent > ENTROPY_THRESH * 0.8 and
                         avg_var > VAR_THRESH * 0.8 and
                         not is_abnormal)

        if is_abnormal and cooldown == 0:
            alert_count += 1
            print(f"\n ALERT at frame {frame_idx} | "
                  f"entropy={ent:.3f} | dirvar={var:.3f}\n")
            cooldown = 25

        if cooldown > 0:
            cooldown -= 1

        # Status
        if is_abnormal:
            status, color = "ABNORMAL", (0, 0, 255)
        elif early_warning:
            status, color = "WARNING",  (0, 165, 255)
        else:
            status, color = "NORMAL",   (0, 255, 0)

        print(f"{frame_idx:>6} | {ent:>8.4f} | {var:>8.4f} | "
              f"{avg_ent:>8.4f} | {avg_var:>8.4f} | {status}")

        # ── Display ───────────────────────────────────────────────────────────
        disp = cv2.resize(frame, (854, 480))

        # Metric bars
        bar_w = int(avg_ent * 300)
        cv2.rectangle(disp, (10, 10), (310, 30), (50,50,50), -1)
        cv2.rectangle(disp, (10, 10), (10+bar_w, 30), (0,200,255), -1)
        cv2.putText(disp, f"Entropy: {avg_ent:.3f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        bar_w2 = int(avg_var * 300)
        cv2.rectangle(disp, (10, 38), (310, 58), (50,50,50), -1)
        cv2.rectangle(disp, (10, 38), (10+bar_w2, 58), (0,255,150), -1)
        cv2.putText(disp, f"Dir Var: {avg_var:.3f}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        cv2.putText(disp, f"Frame: {frame_idx}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.putText(disp, f"Alerts: {alert_count}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        # Status overlay
        cv2.putText(disp, status, (600, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Alert flash
        if is_abnormal:
            cv2.rectangle(disp, (0,0), (853,479), (0,0,255), 4)

        cv2.imshow("Crowd Instability Detection", disp)

        mag_seq.pop(0)
        angle_seq.pop(0)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev = frame.copy()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Total alerts: {alert_count}")