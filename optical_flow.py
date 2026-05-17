"""
Module 2 — Motion Extraction Module
=====================================
Input  : Consecutive frame pairs
Output : Optical flow magnitude maps saved as PNG images
Uses   : cv2.calcOpticalFlowFarneback (dense optical flow)
Purpose: Detect and quantify crowd movement patterns
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


# ── Farneback parameters (tuned for crowd surveillance) ──────────────────────
FARNEBACK_PARAMS = dict(
    pyr_scale  = 0.5,   # pyramid scale (< 1 = downscale per level)
    levels     = 3,     # number of pyramid levels
    winsize    = 15,    # averaging window size
    iterations = 3,     # iterations at each level
    poly_n     = 5,     # pixel neighbourhood size
    poly_sigma = 1.2,   # Gaussian std for polynomial expansion
    flags      = 0,
)


def compute_optical_flow(frame_paths: list, output_dir: str = "flow") -> tuple:
    """
    Step 3: Compute dense optical flow (Farneback) for all consecutive frame pairs.

    For each pair (frame_t, frame_t+1):
        - Compute dx, dy  (flow vectors)
        - Convert to magnitude & angle (polar form)
        - Save magnitude image to output_dir

    Args:
        frame_paths : Sorted list of frame file paths.
        output_dir  : Directory to save magnitude PNG images.

    Returns:
        (magnitude_paths, angle_data)
        magnitude_paths : list of saved .png file paths
        angle_data      : list of angle arrays (needed for entropy / variance)
    """
    os.makedirs(output_dir, exist_ok=True)

    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    magnitude_paths = []
    angle_data      = []   # kept in memory for sequence builder

    prev_gray = cv2.cvtColor(cv2.imread(frame_paths[0]), cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(1, len(frame_paths)), desc="Computing optical flow"):
        curr_frame = cv2.imread(frame_paths[i])
        curr_gray  = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Dense optical flow → flow shape: (H, W, 2)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARNEBACK_PARAMS)

        dx, dy = flow[..., 0], flow[..., 1]

        # Convert to polar: magnitude and angle
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

        # Normalise magnitude to [0, 255] for saving
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_uint8 = mag_norm.astype(np.uint8)

        save_path = os.path.join(output_dir, f"flow_{i:04d}.png")
        cv2.imwrite(save_path, mag_uint8)

        magnitude_paths.append(save_path)
        angle_data.append(angle)          # raw angles preserved

        prev_gray = curr_gray

    print(f"[MotionExtraction] Saved {len(magnitude_paths)} flow maps → '{output_dir}/'")
    return magnitude_paths, angle_data


def load_magnitude_map(path: str, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Step 4: Load a saved magnitude PNG and resize to 64×64.

    Args:
        path        : Path to magnitude PNG.
        target_size : (width, height) — default 64×64.

    Returns:
        Resized magnitude map as float32 in [0, 1].
    """
    mag = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mag_resized = cv2.resize(mag, target_size, interpolation=cv2.INTER_LINEAR)
    return mag_resized.astype(np.float32) / 255.0


if __name__ == "__main__":
    import glob, sys

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "frames"
    paths     = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    if not paths:
        print(f"No frames found in '{frame_dir}'. Run frame_extractor.py first.")
    else:
        mag_paths, angles = compute_optical_flow(paths, output_dir="flow")
        print(f"Example magnitude path : {mag_paths[0]}")
        sample = load_magnitude_map(mag_paths[0])
        print(f"Resized map shape       : {sample.shape}  dtype: {sample.dtype}")
