import glob, os

abnormal = sorted(glob.glob("sequences/abnormal/*_mag.npy"))
vid_counts = {}
for f in abnormal:
    parts = os.path.basename(f).split("_")
    vid = parts[0] + "_" + parts[1]
    vid_counts[vid] = vid_counts.get(vid, 0) + 1

top = sorted(vid_counts.items(), key=lambda x: -x[1])[:15]
print("Videos with MOST abnormal sequences (best for demo):")
print("-" * 55)
for vid, count in top:
    scene = vid.split("_")[0]
    num = vid.split("_")[1].lstrip("0") or "0"
    avi = scene + "_" + num.zfill(3) + ".avi"
    exists = os.path.exists("data/videos/" + avi)
    status = "EXISTS" if exists else "MISSING"
    print(f"  {avi:15s}  {count:5d} abnormal seqs  {status}")
