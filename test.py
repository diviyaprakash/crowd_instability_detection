"""
Module 4–8 — Decision / Alert / Signal / Receiver / Action Pipeline
+ Step 12: Evaluation with Accuracy, Precision, Recall, F1, Detection Delay

Usage:
    python test.py --seq_dir sequences --model_path model/best_model.pt
                   [--onset_frame 50]  [--threshold 0.5]

Output format (real-world style):
    Frame 320 — Abnormal detected | Alert sent to control module | WARNING triggered
"""

import os
import argparse
import glob
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset  import CrowdDataset
from cnn_lstm import (
    CrowdInstabilityDetector,
    compute_motion_entropy,
    compute_directional_variance,
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Decision Module
# ─────────────────────────────────────────────────────────────────────────────

def decision_module(prob: float, threshold: float = 0.5) -> int:
    """
    Logic:
        if prob >= threshold  → abnormal (1)
        else                  → normal   (0)
    """
    return 1 if prob >= threshold else 0


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — Alert Module
# ─────────────────────────────────────────────────────────────────────────────

def alert_module(frame_idx: int, prob: float):
    """
    Generates a warning alert string when abnormal crowd motion is detected.
    In production this would trigger a siren, push notification, SMS, etc.
    """
    msg = (
        f"\n{'!'*60}\n"
        f"  ⚠  WARNING: Crowd instability detected!\n"
        f"     Frame : {frame_idx}\n"
        f"     Confidence : {prob*100:.1f}%\n"
        f"{'!'*60}"
    )
    print(msg)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — Signal Transmission Module
# ─────────────────────────────────────────────────────────────────────────────

ALERT_FLAG_FILE = "alert.txt"

def signal_sender(is_abnormal: bool, frame_idx: int):
    """
    Simulate signal transmission.
    For a mini project: write flag to alert.txt (simulated hardware signal).
    In production: use socket / REST API / MQTT.
    """
    signal = 1 if is_abnormal else 0
    with open(ALERT_FLAG_FILE, "w") as f:
        f.write(f"signal={signal}\nframe={frame_idx}\n")

    if is_abnormal:
        print(f"[SignalSender] Alert signal = {signal} sent (frame {frame_idx})")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — Signal Receiver Module
# ─────────────────────────────────────────────────────────────────────────────

def signal_receiver() -> int:
    """
    Reads the alert flag file and returns the signal value.
    Simulates a remote receiver / control room client.
    """
    if not os.path.exists(ALERT_FLAG_FILE):
        return 0
    with open(ALERT_FLAG_FILE) as f:
        for line in f:
            if line.startswith("signal="):
                return int(line.strip().split("=")[1])
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — Action Module
# ─────────────────────────────────────────────────────────────────────────────

def action_module(signal: int, frame_idx: int):
    """
    Real-life actions (simulated in software):
        - Security announcement
        - Open emergency exit gates
        - Stop new entries
        - Sound siren
        - Notify police / control room
    """
    if signal == 1:
        print(f"[ActionModule] Security Alert Activated at frame {frame_idx}")
        print(f"[ActionModule] → Initiating emergency protocol")
        print(f"[ActionModule] → Gate opening command issued")
        print(f"[ActionModule] → Broadcasting crowd management announcement")
    else:
        pass   # Normal — no action required


# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute classification metrics.

    Returns dict with: accuracy, precision, recall, f1
    """
    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
    }


def compute_detection_delay(y_true: list, y_pred: list, onset_frame: int) -> int:
    """
    Detection delay metric.

        delay = detected_frame - onset_frame

    Where detected_frame is the first frame index where the model predicts 1.

    Args:
        y_true      : Ground-truth labels list.
        y_pred      : Predicted labels list.
        onset_frame : Ground-truth frame where abnormal motion starts.

    Returns:
        delay (int): Positive = late detection, Negative = early detection.
                     None if abnormal never detected.
    """
    for i, pred in enumerate(y_pred):
        if pred == 1:
            delay = i - onset_frame
            return delay
    return None   # model never detected abnormal


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  —  Full system pipeline with alert integration
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model      : CrowdInstabilityDetector,
    dataset    : CrowdDataset,
    device     : torch.device,
    threshold  : float = 0.5,
    onset_frame: int   = None,
) -> dict:
    """
    Run the full detection pipeline on every sequence in the dataset.

    Returns evaluation results and alert log.
    """
    model.eval()

    y_true    = []
    y_pred    = []
    y_probs   = []
    alert_log = []

    print(f"\n{'='*60}")
    print(f"  Running inference on {len(dataset)} sequences")
    print(f"  Threshold : {threshold}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for idx in range(len(dataset)):
            seq, label = dataset[idx]
            seq        = seq.unsqueeze(0).to(device)   # (1, T, 1, 64, 64)
            label_val  = int(label.item())

            # Compute motion coherence metrics
            mag_np = seq[:, :, 0, :, :].cpu().numpy()
            ang_np = np.zeros_like(mag_np)

            # Load angle data if available
            if idx < len(dataset.seq_paths):
                angle_path = dataset.seq_paths[idx].replace("_mag.npy", "_angle.npy")
                if os.path.exists(angle_path):
                    ang_np = np.load(angle_path)[np.newaxis]

            ent     = compute_motion_entropy(mag_np)
            var     = compute_directional_variance(ang_np)
            metrics = torch.from_numpy(np.concatenate([ent, var], axis=1)).to(device)

            logit = model(seq, metrics)
            prob  = torch.sigmoid(logit / 3.0).item()   # temperature scaling for calibrated confidence

            # ── Decision Module ───────────────────────────────────────────────
            prediction = decision_module(prob, threshold)

            y_true.append(label_val)
            y_pred.append(prediction)
            y_probs.append(prob)

            # ── Alert Pipeline ────────────────────────────────────────────────
            status = "Abnormal" if prediction == 1 else "Normal"
            print(f"Seq {idx:04d}  |  GT: {label_val}  |  Pred: {prediction}  "
                  f"|  Prob: {prob:.3f}  |  Status: {status}")

            if prediction == 1:
                alert_msg = alert_module(idx, prob)
                signal_sender(is_abnormal=True, frame_idx=idx)

                signal = signal_receiver()
                action_module(signal, frame_idx=idx)

                alert_log.append({
                    "frame"  : idx,
                    "prob"   : prob,
                    "message": alert_msg,
                })

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics_dict = compute_metrics(y_true, y_pred)

    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy  : {metrics_dict['accuracy']:.4f}")
    print(f"  Precision : {metrics_dict['precision']:.4f}")
    print(f"  Recall    : {metrics_dict['recall']:.4f}")
    print(f"  F1 Score  : {metrics_dict['f1']:.4f}")

    if onset_frame is not None:
        delay = compute_detection_delay(y_true, y_pred, onset_frame)
        if delay is not None:
            print(f"  Detection Delay : {delay} sequences  "
                  f"({'late' if delay > 0 else 'early' if delay < 0 else 'exact'})")
        else:
            print("  Detection Delay : N/A (abnormal never detected)")
        metrics_dict["detection_delay"] = delay

    print(f"{'='*60}\n")

    # Clean up signal file after inference
    if os.path.exists(ALERT_FLAG_FILE):
        os.remove(ALERT_FLAG_FILE)

    return {
        "metrics"  : metrics_dict,
        "y_true"   : y_true,
        "y_pred"   : y_pred,
        "y_probs"  : y_probs,
        "alert_log": alert_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Test Crowd Instability Detector")
    parser.add_argument("--seq_dir",    type=str,   default="sequences",        help="Sequences directory")
    parser.add_argument("--model_path", type=str,   default="model/best_model.pt", help="Saved model checkpoint")
    parser.add_argument("--threshold",  type=float, default=0.5,                help="Classification threshold")
    parser.add_argument("--onset_frame",type=int,   default=None,               help="GT abnormal onset frame (for delay metric)")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Device : {device}")

    # Load dataset
    dataset = CrowdDataset(args.seq_dir)

    # Load model
    model = CrowdInstabilityDetector().to(device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Test] Loaded checkpoint from {args.model_path}")
        print(f"       Trained for {ckpt['epoch']} epoch(s)  |  Val acc: {ckpt['val_acc']:.4f}")
    else:
        print(f"[Test] WARNING: No checkpoint at '{args.model_path}'. Using random weights.")

    results = run_inference(
        model       = model,
        dataset     = dataset,
        device      = device,
        threshold   = args.threshold,
        onset_frame = args.onset_frame,
    )

    if results["alert_log"]:
        print(f"[AlertLog] Total abnormal events detected: {len(results['alert_log'])}")
        for event in results["alert_log"]:
            print(f"  → Frame {event['frame']}  Confidence {event['prob']*100:.1f}%")
    else:
        print("[AlertLog] No abnormal events detected.")
