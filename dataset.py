"""
Dataset Loader (PyTorch)
========================
Step 6 — CrowdDataset

Loads pre-built .npy sequences and their labels.
Labels: 0 = normal,  1 = abnormal

Label convention
----------------
Each video folder under data/videos/ is expected to contain a labels.txt
with one integer (0 or 1) per sequence line.  For quick experiments you
can also pass a flat label list directly.

Directory layout expected:
    sequences/
        normal/
            seq_0000_mag.npy
            seq_0001_mag.npy
            …
        abnormal/
            seq_0000_mag.npy
            …

OR a flat directory with a companion labels.txt:
    sequences/
        seq_0000_mag.npy   →   label from labels.txt line 0
        seq_0001_mag.npy   →   label from labels.txt line 1
        …
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CrowdDataset(Dataset):
    """
    PyTorch Dataset for crowd motion sequences.

    Args:
        seq_dir  : Path to sequence directory.
        labels   : List[int] aligned with sorted sequence files.
                   If None, tries to load from seq_dir/labels.txt.
        transform: Optional callable applied to each tensor.
    """

    def __init__(self, seq_dir: str, labels: list = None, transform=None):
        self.transform = transform

        # ── Gather all magnitude .npy files ──────────────────────────────────
        # Support both flat and class-subfolder layouts
        normal_paths   = sorted(glob.glob(os.path.join(seq_dir, "normal",   "*_mag.npy")))
        abnormal_paths = sorted(glob.glob(os.path.join(seq_dir, "abnormal", "*_mag.npy")))

        if normal_paths or abnormal_paths:
            # Sub-folder layout
            self.seq_paths = normal_paths + abnormal_paths
            self.labels    = [0] * len(normal_paths) + [1] * len(abnormal_paths)
        else:
            # Flat layout — use provided labels or labels.txt
            self.seq_paths = sorted(glob.glob(os.path.join(seq_dir, "*_mag.npy")))
            if labels is not None:
                self.labels = labels
            else:
                label_file = os.path.join(seq_dir, "labels.txt")
                if os.path.exists(label_file):
                    with open(label_file) as f:
                        self.labels = [int(l.strip()) for l in f if l.strip()]
                else:
                    # Fallback: all normal (for inference / demo)
                    print("[Dataset] WARNING: No labels found. Defaulting all to 0 (normal).")
                    self.labels = [0] * len(self.seq_paths)

        assert len(self.seq_paths) == len(self.labels), (
            f"Mismatch: {len(self.seq_paths)} sequences vs {len(self.labels)} labels."
        )

        print(f"[Dataset] Loaded {len(self.seq_paths)} sequences "
              f"(normal={self.labels.count(0)}, abnormal={self.labels.count(1)})")

    # ── PyTorch Dataset interface ─────────────────────────────────────────────

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        seq   = np.load(self.seq_paths[idx]).astype(np.float32)   # (16, 64, 64)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Add channel dim → (16, 1, 64, 64) so CNN sees 1-channel magnitude images
        seq_tensor = torch.from_numpy(seq).unsqueeze(1)            # (16, 1, 64, 64)

        if self.transform:
            seq_tensor = self.transform(seq_tensor)

        return seq_tensor, label


# ── Helper: split into train / val dataloaders ────────────────────────────────

def get_dataloaders(
    seq_dir    : str,
    labels     : list = None,
    batch_size : int  = 8,
    val_split  : float = 0.2,
    num_workers: int  = 4,
    seed       : int  = 42,
):
    """
    Returns (train_loader, val_loader) for a CrowdDataset.
    """
    dataset = CrowdDataset(seq_dir, labels=labels)

    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[Dataset] Train : {len(train_set)}  |  Val : {len(val_set)}")
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else "sequences"
    ds      = CrowdDataset(seq_dir)
    if len(ds) > 0:
        seq, lbl = ds[0]
        print(f"Sequence tensor shape : {seq.shape}")
        print(f"Label                 : {lbl.item()}")
