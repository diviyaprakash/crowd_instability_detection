"""
Module 1 — Video Capture Module
================================
Input  : Video file (.avi / .mp4)
Output : Extracted frames saved as JPEG images
Purpose: Surveillance input preprocessing
"""

import cv2
import os
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str = "frames", max_frames: int = None) -> list:
    """
    Step 1 & 2: Load video and extract frames.

    Args:
        video_path  : Path to input video file.
        output_dir  : Directory to save extracted frames.
        max_frames  : Optional cap on number of frames to extract.

    Returns:
        List of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VideoCapture] Video : {video_path}")
    print(f"[VideoCapture] Total frames : {total_frames}  |  FPS : {fps:.2f}")
    print(f"[VideoCapture] Resolution   : {width} x {height}")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    saved_paths = []
    frame_idx   = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Save as frames/frame_0001.jpg
            filename   = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_paths.append(filename)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"[VideoCapture] Saved {len(saved_paths)} frames → '{output_dir}/'")
    return saved_paths


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/sample.avi"
    extract_frames(video, output_dir="frames")
