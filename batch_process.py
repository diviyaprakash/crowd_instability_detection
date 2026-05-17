import os
import glob
import shutil
import numpy as np
import cv2
from tqdm import tqdm

from frame_extractor import extract_frames
from optical_flow import compute_optical_flow
from sequence_builder import build_sequences

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAINING_VIDEO_DIR = "data/videos"
SEQUENCES_NORMAL   = "sequences/normal"
SEQUENCES_ABNORMAL = "sequences/abnormal"
TEST_FRAMES_DIR    = "data/test_frames"
TEST_MASKS_DIR     = "data/test_masks"
MAX_NORMAL_VIDEOS  = 40
SEQ_LEN            = 16

os.makedirs(SEQUENCES_NORMAL,   exist_ok=True)
os.makedirs(SEQUENCES_ABNORMAL, exist_ok=True)
os.makedirs("frames_temp",      exist_ok=True)
os.makedirs("flow_temp",        exist_ok=True)

# ── STEP A: Process 40 normal training videos ─────────────────────────────────
print("\n" + "="*60)
print("STEP A — Processing normal training videos")
print("="*60)

video_paths = sorted(glob.glob(os.path.join(TRAINING_VIDEO_DIR, "*.avi")))[:MAX_NORMAL_VIDEOS]
print(f"Processing {len(video_paths)} videos...")

normal_seq_counter = 0

for vid_path in tqdm(video_paths, desc="Training videos"):
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]

    # Clear temp folders
    for f in glob.glob("frames_temp/*.jpg"): os.remove(f)
    for f in glob.glob("flow_temp/*.png"):   os.remove(f)

    try:
        # Extract frames
        frame_paths = extract_frames(vid_path, output_dir="frames_temp")

        # Optical flow
        mag_paths, angle_data = compute_optical_flow(frame_paths, output_dir="flow_temp")

        # Build sequences
        seq_paths = build_sequences(
            magnitude_dir = "flow_temp",
            angle_data    = angle_data,
            output_dir    = "seq_temp",
            seq_len       = SEQ_LEN,
        )

        # Move sequences to normal folder with unique names
        for sp in seq_paths:
            basename = os.path.basename(sp)
            new_name = f"{vid_name}_{normal_seq_counter:05d}_{basename}"
            shutil.move(sp, os.path.join(SEQUENCES_NORMAL, new_name))

            # Move angle file too
            angle_sp = sp.replace("_mag.npy", "_angle.npy")
            if os.path.exists(angle_sp):
                new_angle = new_name.replace("_mag.npy", "_angle.npy")
                shutil.move(angle_sp, os.path.join(SEQUENCES_NORMAL, new_angle))

            normal_seq_counter += 1

    except Exception as e:
        print(f"  Skipping {vid_name}: {e}")
        continue

print(f"\n✓ Normal sequences created: {normal_seq_counter}")


# ── STEP B: Process test frames for abnormal sequences ────────────────────────
print("\n" + "="*60)
print("STEP B — Processing test videos for abnormal sequences")
print("="*60)

# Test frames are organized as test_frames/video_name/frame_XXXX.jpg
test_video_dirs = sorted(glob.glob(os.path.join(TEST_FRAMES_DIR, "*")))
abnormal_seq_counter = 0

for test_vid_dir in tqdm(test_video_dirs, desc="Test videos"):
    vid_name  = os.path.basename(test_vid_dir)
    mask_path = os.path.join(TEST_MASKS_DIR, f"{vid_name}.npy")

    if not os.path.exists(mask_path):
        continue

    # Load frame-level mask
    mask = np.load(mask_path)   # 1D array: 0=normal, 1=abnormal

    # Clear temp folders
    for f in glob.glob("flow_temp/*.png"): os.remove(f)

    try:
        frame_paths = sorted(glob.glob(os.path.join(test_vid_dir, "*.jpg")))
        if len(frame_paths) < SEQ_LEN:
            continue

        mag_paths, angle_data = compute_optical_flow(frame_paths, output_dir="flow_temp")

        # Build sequences with labels from mask
        N = len(mag_paths)
        for start in range(N - SEQ_LEN + 1):
            end = start + SEQ_LEN

            # Sequence label: abnormal if ANY frame in window is abnormal
            window_mask = mask[start:end] if end <= len(mask) else mask[start:]
            is_abnormal = int(window_mask.any()) if len(window_mask) > 0 else 0

            if is_abnormal == 0:
                continue   # skip normal test sequences — we have enough normal

            # Load and save this sequence
            seq_mag = []
            for i in range(start, end):
                img = cv2.imread(mag_paths[i], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (64, 64))
                seq_mag.append(img.astype(np.float32) / 255.0)

            seq_array = np.stack(seq_mag, axis=0)  # (16, 64, 64)
            save_path = os.path.join(
                SEQUENCES_ABNORMAL,
                f"{vid_name}_{abnormal_seq_counter:05d}_mag.npy"
            )
            np.save(save_path, seq_array)
            abnormal_seq_counter += 1

    except Exception as e:
        print(f"  Skipping {vid_name}: {e}")
        continue

print(f"\n✓ Abnormal sequences created: {abnormal_seq_counter}")
print(f"\nFINAL COUNT:")
print(f"  Normal   : {normal_seq_counter}")
print(f"  Abnormal : {abnormal_seq_counter}")