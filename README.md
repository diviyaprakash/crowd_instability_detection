# Early-Stage Crowd Instability Detection
### Optical Flow + CNN + LSTM  |  PyTorch

---

## Overview

A motion-based deep learning system that detects abnormal crowd motion by analysing
optical flow sequences using a CNN + LSTM architecture with motion coherence metrics.

Designed for surveillance cameras in public places — railway stations, airports,
stadiums, malls, temples — to detect early crowd instability and send warning alerts
to a control room before a stampede occurs.

---

## System Architecture

```
CCTV Camera
    ↓
Video Capture Module        (frame_extractor.py)
    ↓
Motion Extraction Module    (optical_flow.py)
    ↓
Deep Learning Module        (cnn_lstm.py)
    ↓
Decision Module             (test.py → decision_module)
    ↓
Alert Module                (test.py → alert_module)
    ↓
Signal Transmission Module  (test.py → signal_sender)
    ↓
Signal Receiver Module      (test.py → signal_receiver)
    ↓
Action Module               (test.py → action_module)
```

---

## Folder Structure

```
project/
    data/videos/            ← Place input videos here
    frames/                 ← Extracted JPEG frames
    flow/                   ← Optical flow magnitude maps
    sequences/              ← Sliding-window .npy sequences
    model/                  ← Saved model checkpoints
    frame_extractor.py      ← Step 1 & 2
    optical_flow.py         ← Step 3 & 4
    sequence_builder.py     ← Step 5
    dataset.py              ← Step 6
    cnn_lstm.py             ← Steps 7–10
    train.py                ← Step 11
    test.py                 ← Step 12 + Alert modules
    run_pipeline.py         ← Master orchestrator
    requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Full pipeline (one command)

```bash
python run_pipeline.py --video data/videos/crowd.avi \
                       --epochs 15 \
                       --label 1 \
                       --onset_frame 50
```

### Step by step

```bash
# 1. Extract frames
python frame_extractor.py data/videos/crowd.avi

# 2. Compute optical flow
python optical_flow.py frames/

# 3. Build sequences
python sequence_builder.py flow/

# 4. Train
python train.py --seq_dir sequences --epochs 15

# 5. Test + alerts
python test.py --seq_dir sequences \
               --model_path model/best_model.pt \
               --threshold 0.5 \
               --onset_frame 50
```

---

## Pipeline Steps

| Step | Description | File |
|------|-------------|------|
| 1–2  | Load video → extract frames | frame_extractor.py |
| 3    | Farneback dense optical flow → dx, dy → magnitude + angle | optical_flow.py |
| 4    | Resize magnitude map to 64×64 | optical_flow.py |
| 5    | Sliding-window sequences (len=16) | sequence_builder.py |
| 6    | PyTorch CrowdDataset (0=normal, 1=abnormal) | dataset.py |
| 7    | CNN: Conv→ReLU→Pool×2 → 128-dim feature vector | cnn_lstm.py |
| 8    | LSTM: temporal modelling over sequence of CNN features | cnn_lstm.py |
| 9    | Motion entropy + directional variance appended | cnn_lstm.py |
| 10   | Linear+Sigmoid classifier | cnn_lstm.py |
| 11   | Train: BCEWithLogitsLoss + Adam | train.py |
| 12   | Evaluate: Accuracy, Precision, Recall, F1, Detection Delay | test.py |

---

## Model Architecture

```
Input: (B, 16, 1, 64, 64)
         ↓
CNN per frame → (B, 16, 128)
         ↓
LSTM → (B, 256)
         ↓
Concat [entropy, directional_var] → (B, 258)
         ↓
Linear(258→64) → ReLU → Linear(64→1)
         ↓
BCEWithLogitsLoss (training) / Sigmoid (inference)
         ↓
Output: Normal (0) / Abnormal (1)
```

---

## Evaluation Metrics

- **Accuracy** — overall correct classifications
- **Precision** — of abnormal predictions, how many were correct
- **Recall** — of actual abnormal events, how many were caught
- **F1** — harmonic mean of precision and recall
- **Detection Delay** — `detected_frame − onset_frame`
  - Positive → late detection
  - Negative → early detection (false alarm)
  - 0 → perfect timing

---

## Alert System Output

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ⚠  WARNING: Crowd instability detected!
     Frame : 320
     Confidence : 87.3%
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

[SignalSender]  Alert signal = 1 sent (frame 320)
[ActionModule]  Security Alert Activated at frame 320
[ActionModule]  → Initiating emergency protocol
[ActionModule]  → Gate opening command issued
[ActionModule]  → Broadcasting crowd management announcement
```

---

## Module Definitions

| Module | Purpose | Output |
|--------|---------|--------|
| Video Capture | Read CCTV stream, save frames | JPEG frames |
| Motion Extraction | Farneback optical flow | Magnitude maps |
| Deep Learning | CNN + LSTM detection | Normal / Abnormal |
| Decision | Threshold-based classification | Alert signal |
| Alert | Generate warning message | Warning text |
| Signal Transmission | Write flag to alert.txt | signal=1 file |
| Signal Receiver | Read alert flag | Integer signal |
| Action | Trigger security response | Console / API call |

---

## Dataset

This project was trained and evaluated using the ShanghaiTech Crowd Dataset.

The system analyses surveillance crowd videos to detect abnormal crowd motion
and early-stage instability using optical flow and deep learning.

Supported input formats:
- .avi
- .mp4

Dataset files are not included in this repository due to GitHub storage limitations.

Label structure:

```
sequences/
    normal/        ← all normal sequences here
    abnormal/      ← all abnormal sequences here
```

Or flat with labels.txt:

```
sequences/
    seq_0000_mag.npy
    seq_0001_mag.npy
    labels.txt        ← one integer per line (0 or 1)
```

---

## Base Paper

Direkoğlu, C. (2020). *Motion-Based Crowd Instability Detection*. IEEE Access.

**Extensions added in this project:**
- LSTM for temporal modelling
- Motion entropy per sequence
- Directional variance per sequence
- Detection delay metric
- Full alert signal pipeline (Modules 4–8)
