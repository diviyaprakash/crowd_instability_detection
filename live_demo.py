"""
Live Video Demo — Real-time Crowd Instability Detection
========================================================
Processes any video file frame-by-frame with live overlay showing:
  - Per-sequence probability
  - Rolling average
  - Early WARNING (rising trend)
  - ALERT (confirmed abnormal)

Usage:
    python live_demo.py --video data/videos/01_053.avi
    python live_demo.py --video path/to/any_crowd_video.mp4
"""

import cv2
import torch
import numpy as np
import argparse
from collections import deque

from cnn_lstm import (
    CrowdInstabilityDetector,
    compute_motion_entropy,
    compute_directional_variance,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "model/best_model.pt"
SEQ_LEN      = 16
TEMPERATURE  = 3.0       # soften overconfident predictions
THRESHOLD    = 0.5
WINDOW       = 5         # rolling window for averaging
CONFIRM_COUNT = 3        # consecutive high-prob frames to confirm alert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = CrowdInstabilityDetector(
        seq_len=SEQ_LEN, feature_dim=128, hidden_dim=256, num_layers=2
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Model] Loaded from epoch {ckpt['epoch']} | val_acc={ckpt['val_acc']:.4f} | device={DEVICE}")
    return model


def compute_flow_magnitude(prev_gray, curr_gray):
    """Compute Farneback optical flow magnitude, normalized to [0,1] and resized to 64x64.
    
    Matches training pipeline: optical_flow.py normalizes to [0,255] as uint8,
    then load_magnitude_map() divides by 255 → [0,1].
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude to [0,1] — same as training pipeline
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_norm = mag_norm.astype(np.float32) / 255.0
    
    mag_resized = cv2.resize(mag_norm, (64, 64))
    ang_resized = cv2.resize(ang, (64, 64))
    return mag_resized, ang_resized


def is_rising_trend(history):
    """Check if last 3 values show a rising trend."""
    if len(history) < 3:
        return False
    h = list(history)
    return h[-1] > h[-2] > h[-3]


def draw_overlay(frame, prob, avg_prob, status, color, seq_count, alert_count):
    """Draw HUD overlay on frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status indicator
    cv2.putText(frame, status, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Probability bar
    bar_x, bar_y, bar_w, bar_h = 15, 50, 250, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill_w = int(bar_w * min(prob, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"Prob: {prob:.3f}", (bar_x + bar_w + 10, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Rolling average bar
    bar_y2 = bar_y + 20
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (40, 40, 40), -1)
    fill_w2 = int(bar_w * min(avg_prob, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill_w2, bar_y2 + bar_h), (0, 180, 255), -1)
    cv2.putText(frame, f"Avg:  {avg_prob:.3f}", (bar_x + bar_w + 10, bar_y2 + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Counters
    cv2.putText(frame, f"SEQ: {seq_count}", (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)
    cv2.putText(frame, f"ALERTS: {alert_count}", (w - 180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if alert_count > 0 else (100, 100, 100), 1)

    # Bottom bar — system title
    cv2.rectangle(frame, (0, h - 30), (w, h), (10, 10, 10), -1)
    cv2.putText(frame, "CROWD INSTABILITY DETECTION SYSTEM  |  CNN-LSTM  |  Farneback Optical Flow",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)


def main(video_path, threshold):
    model = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {video_path} | {total_frames} frames | {fps:.0f} FPS")

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    mag_buffer = []       # stores magnitude frames for building sequences
    ang_buffer = []       # stores angle frames
    prob_history = deque(maxlen=WINDOW)
    cooldown = 0
    seq_count = 0
    alert_count = 0
    frame_idx = 0

    print(f"\n[Running] Threshold={threshold} | Press ESC to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        mag, ang = compute_flow_magnitude(prev_gray, curr_gray)
        mag_buffer.append(mag)
        ang_buffer.append(ang)

        # Default status
        prob = 0.0
        avg_prob = 0.0
        status = "SCANNING..."
        color = (150, 150, 150)

        # When we have enough frames for a sequence
        if len(mag_buffer) >= SEQ_LEN:
            seq_mag = np.array(mag_buffer[-SEQ_LEN:], dtype=np.float32)    # (16, 64, 64)
            seq_ang = np.array(ang_buffer[-SEQ_LEN:], dtype=np.float32)    # (16, 64, 64)

            # Prepare tensor: (1, 16, 1, 64, 64)
            seq_tensor = torch.from_numpy(seq_mag).unsqueeze(0).unsqueeze(2).to(DEVICE)

            # Compute motion metrics
            mag_np = seq_mag[np.newaxis]     # (1, 16, 64, 64)
            ang_np = seq_ang[np.newaxis]     # (1, 16, 64, 64)
            ent = compute_motion_entropy(mag_np)
            var = compute_directional_variance(ang_np)
            metrics = torch.from_numpy(
                np.concatenate([ent, var], axis=1)
            ).to(DEVICE)

            # Inference
            with torch.no_grad():
                logit = model(seq_tensor, metrics)
                prob = torch.sigmoid(logit / TEMPERATURE).item()

            seq_count += 1
            prob_history.append(prob)
            avg_prob = float(np.mean(prob_history))

            # ── Decision Logic ────────────────────────────────────────────
            high_count = sum(1 for p in prob_history if p > threshold)
            is_alert = avg_prob > threshold and high_count >= CONFIRM_COUNT
            early_warning = is_rising_trend(prob_history) and avg_prob > threshold * 0.6

            if is_alert:
                status = "ALERT - ABNORMAL"
                color = (0, 0, 255)      # Red
                if cooldown == 0:
                    alert_count += 1
                    print(f"  [ALERT] Frame {frame_idx} | prob={prob:.3f} | avg={avg_prob:.3f}")
                    cooldown = 20
            elif early_warning:
                status = "WARNING - BUILDUP"
                color = (0, 165, 255)    # Orange
                if cooldown == 0:
                    print(f"  [WARN]  Frame {frame_idx} | prob={prob:.3f} | avg={avg_prob:.3f} | rising trend")
            else:
                status = "NORMAL"
                color = (0, 255, 0)      # Green

            if cooldown > 0:
                cooldown -= 1

            # Slide window by 1 frame (not full SEQ_LEN)
            mag_buffer.pop(0)
            ang_buffer.pop(0)

        # Draw overlay
        draw_overlay(frame, prob, avg_prob, status, color, seq_count, alert_count)

        # Show frame
        cv2.imshow("Crowd Instability Detection", frame)

        # ESC to quit, or slow down to ~original FPS
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == 27:
            break

        prev_gray = curr_gray

    cap.release()
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*50}")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Sequences analyzed: {seq_count}")
    print(f"  Alerts triggered  : {alert_count}")
    print(f"  Threshold         : {threshold}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live Crowd Instability Detection Demo")
    p.add_argument("--video", type=str, required=True, help="Path to video file")
    p.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = p.parse_args()
    main(args.video, args.threshold)
