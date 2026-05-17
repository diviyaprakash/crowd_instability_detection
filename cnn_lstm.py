"""
Module 3 — Deep Learning Detection Module
==========================================
Architecture: CNN → LSTM → Entropy+Variance → Classifier

Step 7  — CNN   : spatial feature extractor per frame (64×64 → 128-dim vector)
Step 8  — LSTM  : temporal sequence modelling
Step 9  — Metrics: motion entropy + directional variance appended to features
Step 10 — Classifier: Linear → Sigmoid → Normal / Abnormal
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  —  Spatial CNN Feature Extractor
# Input  : (B, 1, 64, 64)  single-channel magnitude map
# Output : (B, 128)         feature vector
# ─────────────────────────────────────────────────────────────────────────────

class MotionCNN(nn.Module):
    """
    Lightweight CNN for extracting spatial motion features from a single
    64×64 magnitude map.

    Architecture:
        Conv(1→32)  → ReLU → MaxPool(2)   →  32×32
        Conv(32→64) → ReLU → MaxPool(2)   →  16×16
        Conv(64→128)→ ReLU → AdaptiveAvgPool → 1×1
        Flatten → Linear → 128-dim feature
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32×32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 16×16

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                 # 1×1
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, x):
        # x: (B, 1, 64, 64)
        return self.fc(self.conv_block(x))   # → (B, feature_dim)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  —  Temporal LSTM
# Input  : (B, T, feature_dim)  sequence of CNN features
# Output : (B, hidden_dim)      temporal representation of the whole sequence
# ─────────────────────────────────────────────────────────────────────────────

class TemporalLSTM(nn.Module):
    """
    Single-layer LSTM that reads a sequence of CNN feature vectors and
    returns the hidden state after the last time step.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden_dim)
        return h_n[-1]               # last layer's hidden state → (B, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  —  Motion Coherence Metrics (NumPy, runs outside autograd)
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion_entropy(magnitude_seq: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Compute per-sequence motion entropy from magnitude histograms.

    Args:
        magnitude_seq: (B, T, H, W)  batch of magnitude sequences, values in [0,1]
        bins         : histogram bins for entropy computation

    Returns:
        entropy: (B, 1)  normalised Shannon entropy
    """
    B, T, H, W = magnitude_seq.shape
    entropies = []
    for b in range(B):
        seq_entropy = []
        for t in range(T):
            frame = magnitude_seq[b, t].ravel()
            hist, _ = np.histogram(frame, bins=bins, range=(0, 1), density=True)
            hist    = hist + 1e-9          # avoid log(0)
            hist    = hist / hist.sum()
            ent     = -np.sum(hist * np.log2(hist))
            seq_entropy.append(ent)
        entropies.append(np.mean(seq_entropy))

    max_ent = np.log2(bins)
    return (np.array(entropies) / max_ent).reshape(B, 1).astype(np.float32)


def compute_directional_variance(angle_seq: np.ndarray) -> np.ndarray:
    """
    Compute per-sequence directional variance from flow angle sequences.
    High variance → chaotic / abnormal motion.

    Args:
        angle_seq: (B, T, H, W)  angles in degrees [0, 360)

    Returns:
        var: (B, 1)  normalised directional variance in [0, 1]
    """
    B, T, H, W = angle_seq.shape
    variances = []
    for b in range(B):
        rad = np.deg2rad(angle_seq[b])   # (T, H, W)
        sin_mean = np.mean(np.sin(rad))
        cos_mean = np.mean(np.cos(rad))
        R        = np.sqrt(sin_mean**2 + cos_mean**2)   # resultant length ∈ [0,1]
        var      = 1.0 - R                               # high when directions scattered
        variances.append(var)

    return np.array(variances).reshape(B, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10  —  Full CNN-LSTM Classifier
# ─────────────────────────────────────────────────────────────────────────────

class CrowdInstabilityDetector(nn.Module):
    """
    End-to-end model:
        1. CNN encodes each frame in the sequence independently.
        2. LSTM models temporal dynamics across the sequence.
        3. Motion coherence metrics (entropy + directional variance) are
           concatenated to the LSTM output.
        4. Linear classifier → scalar logit → BCEWithLogitsLoss.

    Args:
        seq_len      : Number of frames per sequence (default 16).
        feature_dim  : CNN output dimension (default 128).
        hidden_dim   : LSTM hidden size (default 256).
        num_layers   : LSTM layers (default 2).
        metric_dim   : Extra metric features appended (entropy + var = 2).
    """

    def __init__(
        self,
        seq_len    : int = 16,
        feature_dim: int = 128,
        hidden_dim : int = 256,
        num_layers : int = 2,
        metric_dim : int = 2,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.feature_dim = feature_dim

        self.cnn  = MotionCNN(feature_dim=feature_dim)
        self.lstm = TemporalLSTM(input_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)

        # Classifier head: hidden_dim + metrics → 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + metric_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
            # NOTE: no Sigmoid here — BCEWithLogitsLoss expects raw logits
        )

    def encode_sequence(self, x):
        """
        Encode a batch of sequences through the CNN.

        Args:
            x: (B, T, 1, 64, 64)

        Returns:
            cnn_features: (B, T, feature_dim)
        """
        B, T, C, H, W = x.shape
        # Flatten time into batch for efficient CNN processing
        x_flat    = x.view(B * T, C, H, W)                 # (B*T, 1, 64, 64)
        feats_flat = self.cnn(x_flat)                       # (B*T, feature_dim)
        return feats_flat.view(B, T, self.feature_dim)      # (B, T, feature_dim)

    def forward(self, x, metrics: torch.Tensor = None):
        """
        Args:
            x       : (B, T, 1, 64, 64)  magnitude sequence batch
            metrics : (B, 2)             pre-computed [entropy, directional_var]
                      If None, zeros are used (metrics computed externally).

        Returns:
            logits: (B,) raw classification logits
        """
        B = x.size(0)

        cnn_feats    = self.encode_sequence(x)          # (B, T, feature_dim)
        temporal_rep = self.lstm(cnn_feats)             # (B, hidden_dim)

        if metrics is None:
            metrics = torch.zeros(B, 2, device=x.device)

        combined = torch.cat([temporal_rep, metrics], dim=1)   # (B, hidden+2)
        logits   = self.classifier(combined).squeeze(1)        # (B,)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, H, W = 4, 16, 64, 64

    model   = CrowdInstabilityDetector()
    dummy_x = torch.randn(B, T, 1, H, W)

    # Simulate external metrics
    mag_np   = dummy_x[:, :, 0, :, :].numpy()   # (B, T, 64, 64)
    ang_np   = np.random.uniform(0, 360, (B, T, H, W)).astype(np.float32)

    ent = compute_motion_entropy(mag_np)
    var = compute_directional_variance(ang_np)
    metrics = torch.from_numpy(np.concatenate([ent, var], axis=1))

    logits = model(dummy_x, metrics)
    probs  = torch.sigmoid(logits)

    print(f"Model input  : {dummy_x.shape}")
    print(f"Logits shape : {logits.shape}  →  {logits.detach().numpy().round(3)}")
    print(f"Probs        : {probs.detach().numpy().round(3)}")
    print("CNN-LSTM model OK ✓")
