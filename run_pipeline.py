"""
run_pipeline.py — Master Orchestrator
======================================
Ties together all modules in the exact order specified:

    Step 1–2  : Frame extraction
    Step 3–4  : Optical flow + resize
    Step 5    : Sequence creation
    Step 6    : Dataset loading
    Step 11   : Training
    Step 12   : Evaluation + alerts

Usage:
    python run_pipeline.py --video data/videos/sample.avi
                           [--epochs 15] [--threshold 0.5] [--onset_frame 50]
"""

import os
import argparse
import glob
import numpy as np
import torch

from frame_extractor  import extract_frames
from optical_flow     import compute_optical_flow
from sequence_builder import build_sequences
from dataset          import get_dataloaders, CrowdDataset
from cnn_lstm         import CrowdInstabilityDetector
from train            import train as train_model
from test             import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Crowd Instability Detection — Full Pipeline")
    parser.add_argument("--video",       type=str, required=True,  help="Input video file")
    parser.add_argument("--epochs",      type=int, default=15,     help="Training epochs")
    parser.add_argument("--batch_size",  type=int, default=8,      help="Batch size")
    parser.add_argument("--threshold",   type=float, default=0.5,  help="Decision threshold")
    parser.add_argument("--onset_frame", type=int, default=None,   help="Abnormal onset frame for delay metric")
    parser.add_argument("--label",       type=int, default=0,      help="Label for this video: 0=normal, 1=abnormal")
    parser.add_argument("--max_frames",  type=int, default=None,   help="Cap on frames to extract")
    return parser.parse_args()


def run_pipeline(args):
    print("\n" + "="*70)
    print("  CROWD INSTABILITY DETECTION SYSTEM")
    print("  Early-Stage Detection via Optical Flow + CNN + LSTM")
    print("="*70 + "\n")

    # ── STEP 1 & 2: Frame Extraction ─────────────────────────────────────────
    print("► [Module 1] Video Capture — Extracting frames …")
    frame_paths = extract_frames(args.video, output_dir="frames", max_frames=args.max_frames)
    print(f"  ✓ {len(frame_paths)} frames extracted\n")

    # ── STEP 3 & 4: Optical Flow ─────────────────────────────────────────────
    print("► [Module 2] Motion Extraction — Computing Farneback optical flow …")
    mag_paths, angle_data = compute_optical_flow(frame_paths, output_dir="flow")
    print(f"  ✓ {len(mag_paths)} flow magnitude maps computed\n")

    # ── STEP 5: Sequence Creation ─────────────────────────────────────────────
    print("► [SequenceBuilder] Creating sliding-window sequences (len=16) …")
    seq_paths = build_sequences(
        magnitude_dir = "flow",
        angle_data    = angle_data,
        output_dir    = "sequences",
        seq_len       = 16,
    )
    print(f"  ✓ {len(seq_paths)} sequences created\n")

    if len(seq_paths) == 0:
        print("ERROR: No sequences created. Need at least 16 optical flow maps.")
        print(f"       Frames extracted: {len(frame_paths)}  |  Flow maps: {len(mag_paths)}")
        return

    # Write labels file (single video → all sequences get same label)
    label_file = "sequences/labels.txt"
    with open(label_file, "w") as f:
        for _ in seq_paths:
            f.write(f"{args.label}\n")
    print(f"  ✓ Labels written to {label_file}  (all = {args.label})\n")

    # ── STEP 6: Dataset ───────────────────────────────────────────────────────
    print("► [Module 3] Deep Learning — Building dataset …")

    # ── STEP 11: Training ─────────────────────────────────────────────────────
    print("► Training CNN-LSTM model …")

    # Minimal args namespace for train()
    class TrainArgs:
        seq_dir    = "sequences"
        epochs     = args.epochs
        batch_size = args.batch_size
        lr         = 1e-3
        val_split  = 0.2

    trained_model, history = train_model(TrainArgs())
    print("  ✓ Training complete\n")

    # ── STEP 12: Evaluation + Alert Pipeline ─────────────────────────────────
    print("► [Module 4–8] Running full evaluation + alert pipeline …")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CrowdDataset("sequences")

    results = run_inference(
        model       = trained_model,
        dataset     = dataset,
        device      = device,
        threshold   = args.threshold,
        onset_frame = args.onset_frame,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    m = results["metrics"]
    print("\n" + "="*70)
    print("  FINAL PIPELINE SUMMARY")
    print("="*70)
    print(f"  Video          : {args.video}")
    print(f"  Frames         : {len(frame_paths)}")
    print(f"  Flow maps      : {len(mag_paths)}")
    print(f"  Sequences      : {len(seq_paths)}")
    print(f"  Model          : CNN-LSTM  (128-dim CNN, 256-dim LSTM)")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  Recall         : {m['recall']:.4f}")
    print(f"  F1 Score       : {m['f1']:.4f}")
    if "detection_delay" in m and m["detection_delay"] is not None:
        print(f"  Detection Delay: {m['detection_delay']} sequences")
    print(f"  Alerts Raised  : {len(results['alert_log'])}")
    print("="*70 + "\n")

    print("System: This pipeline is designed for surveillance cameras installed")
    print("        in public places (railway stations, airports, stadiums, malls)")
    print("        to detect early crowd instability and send warning alerts to")
    print("        the control room.")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
