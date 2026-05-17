"""
Sequence Creator
=================
Input  : Optical flow magnitude maps (64×64 each)
Output : Sliding-window sequences saved as .npy files
         Shape per file: (SEQ_LEN, 64, 64)

Sliding window:
    [1–16], [2–17], [3–18], …

Also stores angle sequences alongside magnitude sequences
for computing motion coherence metrics later.
"""

import numpy as np
import os
import glob
from tqdm import tqdm

from optical_flow import load_magnitude_map

SEQ_LEN     = 16          # sequence length (frames)
FRAME_SIZE  = (64, 64)    # spatial resolution


def build_sequences(
    magnitude_dir : str,
    angle_data    : list,
    output_dir    : str  = "sequences",
    seq_len       : int  = SEQ_LEN,
) -> list:
    """
    Step 5: Create sliding-window sequences from magnitude maps.

    Args:
        magnitude_dir : Directory containing flow_XXXX.png files.
        angle_data    : List of angle arrays (from optical_flow.compute_optical_flow).
        output_dir    : Where to save .npy sequence files.
        seq_len       : Sliding window length.

    Returns:
        List of saved .npy file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    mag_paths = sorted(glob.glob(os.path.join(magnitude_dir, "flow_*.png")))
    if len(mag_paths) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} flow maps, found {len(mag_paths)}."
        )

    # Pre-load all magnitude maps into memory (resized to 64×64)
    print("[SequenceBuilder] Loading magnitude maps …")
    magnitudes = np.stack(
        [load_magnitude_map(p, target_size=FRAME_SIZE) for p in tqdm(mag_paths)],
        axis=0
    )   # shape: (N, 64, 64)

    # Resize angle arrays to 64×64 as well
    import cv2
    angles_resized = np.stack(
        [cv2.resize(a, FRAME_SIZE, interpolation=cv2.INTER_LINEAR) for a in tqdm(angle_data, desc="Resizing angles")],
        axis=0
    )   # shape: (N, 64, 64)

    N = magnitudes.shape[0]
    saved_paths = []
    seq_idx     = 0

    for start in tqdm(range(N - seq_len + 1), desc="Building sequences"):
        end = start + seq_len

        seq_mag   = magnitudes[start:end]      # (16, 64, 64)
        seq_angle = angles_resized[start:end]  # (16, 64, 64)

        mag_path   = os.path.join(output_dir, f"seq_{seq_idx:04d}_mag.npy")
        angle_path = os.path.join(output_dir, f"seq_{seq_idx:04d}_angle.npy")

        np.save(mag_path,   seq_mag.astype(np.float32))
        np.save(angle_path, seq_angle.astype(np.float32))

        saved_paths.append(mag_path)
        seq_idx += 1

    print(f"[SequenceBuilder] Created {len(saved_paths)} sequences → '{output_dir}/'")
    print(f"[SequenceBuilder] Each sequence shape : {seq_mag.shape}")
    return saved_paths


if __name__ == "__main__":
    import sys
    flow_dir = sys.argv[1] if len(sys.argv) > 1 else "flow"

    # Dummy angle data for standalone testing
    mag_paths = sorted(glob.glob(os.path.join(flow_dir, "flow_*.png")))
    dummy_angles = [np.zeros((480, 640)) for _ in mag_paths]

    seqs = build_sequences(flow_dir, dummy_angles, output_dir="sequences")
    print(f"First sequence path : {seqs[0]}")
    arr  = np.load(seqs[0])
    print(f"Loaded shape        : {arr.shape}  dtype: {arr.dtype}")
