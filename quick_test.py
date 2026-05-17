"""Quick model sanity check — test 10 normal + 10 abnormal sequences."""
import torch, numpy as np, glob, os
from cnn_lstm import CrowdInstabilityDetector, compute_motion_entropy, compute_directional_variance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrowdInstabilityDetector().to(device)
ckpt = torch.load("model/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model from epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

normal_files = sorted(glob.glob("sequences/normal/*_mag.npy"))[:10]
abnormal_files = sorted(glob.glob("sequences/abnormal/*_mag.npy"))[:10]

correct = 0
total = 0

print(f"\n--- Testing 10 NORMAL sequences (should predict NORMAL) ---")
for f in normal_files:
    seq = np.load(f).astype(np.float32)
    seq_t = torch.from_numpy(seq).unsqueeze(0).unsqueeze(2).to(device)
    mag_np = seq_t[:, :, 0, :, :].cpu().numpy()
    ang_np = np.zeros_like(mag_np)
    ent = compute_motion_entropy(mag_np)
    var = compute_directional_variance(ang_np)
    metrics = torch.from_numpy(np.concatenate([ent, var], axis=1)).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(seq_t, metrics)).item()
    pred = "ABNORMAL" if prob >= 0.5 else "NORMAL"
    ok = "OK" if pred == "NORMAL" else "XX"
    correct += 1 if pred == "NORMAL" else 0
    total += 1
    print(f"  {ok} {os.path.basename(f):30s}  prob={prob:.4f}  -> {pred}")

print(f"\n--- Testing 10 ABNORMAL sequences (should predict ABNORMAL) ---")
for f in abnormal_files:
    seq = np.load(f).astype(np.float32)
    seq_t = torch.from_numpy(seq).unsqueeze(0).unsqueeze(2).to(device)
    mag_np = seq_t[:, :, 0, :, :].cpu().numpy()
    ang_np = np.zeros_like(mag_np)
    ent = compute_motion_entropy(mag_np)
    var = compute_directional_variance(ang_np)
    metrics = torch.from_numpy(np.concatenate([ent, var], axis=1)).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(seq_t, metrics)).item()
    pred = "ABNORMAL" if prob >= 0.5 else "NORMAL"
    ok = "OK" if pred == "ABNORMAL" else "XX"
    correct += 1 if pred == "ABNORMAL" else 0
    total += 1
    print(f"  {ok} {os.path.basename(f):30s}  prob={prob:.4f}  -> {pred}")

print(f"\n--- Result: {correct}/{total} correct ({correct/total*100:.0f}%) ---")
