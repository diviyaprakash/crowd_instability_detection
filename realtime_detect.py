import cv2
import torch
import numpy as np
from collections import deque
from cnn_lstm import CrowdInstabilityDetector, compute_motion_entropy, compute_directional_variance

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model/best_model.pt"
VIDEO_PATH = "D:\Minor_Project\Delhi Metro Crowd #shorts.mp4"   # change to your test video
SEQ_LEN    = 16
THRESHOLD  = 0.80    # higher = fewer false positives
WINDOW     = 5
CONFIRM    = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model = CrowdInstabilityDetector().to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded  device={DEVICE}")

# ── UTILS ─────────────────────────────────────────────────────────────────────
def compute_flow(prev, curr):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow      = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag    = cv2.resize(mag, (64, 64))
    return mag.astype(np.float32) / (mag.max() + 1e-8)   # normalise

def is_rising_trend(history):
    if len(history) < 3:
        return False
    return history[-1] > history[-2] > history[-3]

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev_frame = cap.read()

sequence     = []
prob_history = deque(maxlen=WINDOW)
cooldown     = 0
frame_idx    = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mag = compute_flow(prev_frame, frame)
    sequence.append(mag)

    if len(sequence) == SEQ_LEN:
        # Build tensor (1, 16, 1, 64, 64)
        seq_np  = np.stack(sequence, axis=0)              # (16, 64, 64)
        seq_t   = torch.from_numpy(seq_np).unsqueeze(0).unsqueeze(2).to(DEVICE)

        # Compute metrics
        mag_np  = seq_np[np.newaxis]                      # (1, 16, 64, 64)
        ang_np  = np.zeros_like(mag_np)
        ent     = compute_motion_entropy(mag_np)
        var     = compute_directional_variance(ang_np)
        metrics = torch.from_numpy(
            np.concatenate([ent, var], axis=1)
        ).to(DEVICE)

        with torch.no_grad():
            logit = model(seq_t, metrics)
            prob  = torch.sigmoid(logit).item()

        # Decision logic
        prob_history.append(prob)
        avg_prob   = float(np.mean(prob_history))
        high_count = sum(p > THRESHOLD for p in prob_history)

        alert         = avg_prob > THRESHOLD and high_count >= CONFIRM
        early_warning = is_rising_trend(list(prob_history)) and avg_prob > 0.6

        # Cooldown
        if alert and cooldown == 0:
            print(f"ALERT at frame {frame_idx} | prob={prob:.3f} | avg={avg_prob:.3f}")
            cooldown = 20
        if cooldown > 0:
            cooldown -= 1

        # Display
        if alert:
            status, color = "ALERT",   (0, 0, 255)
        elif early_warning:
            status, color = "WARNING", (0, 165, 255)
        else:
            status, color = "NORMAL",  (0, 255, 0)

        cv2.putText(frame, f"Prob:{prob:.2f} Avg:{avg_prob:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        print(f"Frame {frame_idx:04d} | Prob:{prob:.3f} | Avg:{avg_prob:.3f} | {status}")
        sequence.pop(0)

    cv2.imshow("Crowd Instability Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_frame = frame.copy()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()