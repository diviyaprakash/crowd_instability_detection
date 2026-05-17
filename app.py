"""
app.py — Flask Localhost Deployment (Metric-Based Engine)
==========================================================
Detection: Motion Entropy + Directional Variance (rule-based)
Run with : python app.py
Open     : http://localhost:5000
"""

import os, glob, time, json, threading, uuid
import numpy as np
import cv2
from collections import deque
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder=".")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)

ENTROPY_THRESH = 0.50
VAR_THRESH     = 0.35
SEQ_LEN        = 16
WINDOW         = 8
CONFIRM        = 3

processing_state = {
    "running": False, "events": [], "stage": "idle",
    "alert_count": 0, "log": [], "stop_requested": False,
}
state_lock = threading.RLock()
pipeline_thread = None
current_run_id = None

def push_event(t, data):
    with state_lock:
        processing_state["events"].append(
            {"type": t, "data": data, "time": time.strftime("%H:%M:%S")}
        )

def add_log(msg, level="info"):
    with state_lock:
        processing_state["log"].append(
            {"time": time.strftime("%H:%M:%S"), "message": msg, "level": level}
        )

def compute_flow(prev, curr):
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, angle = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    return (cv2.resize(mag,   (64,64)).astype(np.float32),
            cv2.resize(angle, (64,64)).astype(np.float32))

def motion_entropy(mag_seq):
    flat = np.stack(mag_seq).ravel()
    norm = flat / (flat.max() + 1e-8)
    h, _ = np.histogram(norm, bins=32, range=(0,1), density=True)
    h = (h + 1e-9); h /= h.sum()
    return float(-np.sum(h * np.log2(h)) / np.log2(32))

def directional_variance(angle_seq):
    rad = np.deg2rad(np.stack(angle_seq))
    R   = np.sqrt(np.mean(np.sin(rad))**2 + np.mean(np.cos(rad))**2)
    return float(1.0 - R)

def is_rising(hist):
    return len(hist) >= 3 and hist[-1] > hist[-2] > hist[-3]

def run_pipeline(video_path, ent_thr, var_thr, run_id):
    try:
        time.sleep(1.5) # Wait for client to connect to SSE
        add_log("Module 1 — Video Capture", "info")
        push_event("stage", {"stage": "extracting", "message": "Opening video..."})

        cap = cv2.VideoCapture(video_path)
            
        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        is_live = (total <= 0)
        
        if is_live:
            add_log(f"Connected to LIVE stream: {video_path}", "success")
        else:
            add_log(f"Video: {total} frames @ {fps:.1f} FPS", "success")
            
        push_event("progress", {"total": total, "fps": fps, "is_live": is_live})

        ret, prev = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")

        add_log("Module 2 — Farneback Optical Flow", "info")
        add_log("Module 3 — Entropy + Variance Detection", "info")
        push_event("stage", {"stage": "inference", "message": "Analysing crowd motion..."})

        mag_seq = []; angle_seq = []
        ent_h = deque(maxlen=WINDOW); var_h = deque(maxlen=WINDOW)
        cooldown = 0; fidx = 0; alerts = 0
        n_norm = 0; n_warn = 0; n_abn = 0

        while True:
            with state_lock:
                if current_run_id != run_id:
                    add_log("Pipeline overtaken by new run, exiting old thread", "warning")
                    break
                if processing_state.get("stop_requested"):
                    add_log("Pipeline stopped by operator", "warning")
                    break
                    
            ret, frame = cap.read()
            if not ret: break

            mag, angle = compute_flow(prev, frame)
            mag_seq.append(mag); angle_seq.append(angle)

            if len(mag_seq) == SEQ_LEN:
                ent = motion_entropy(mag_seq)
                var = directional_variance(angle_seq)
                ent_h.append(ent); var_h.append(var)

                avg_e = float(np.mean(ent_h))
                avg_v = float(np.mean(var_h))
                h_ent = sum(e > ent_thr for e in ent_h)
                h_var = sum(v > var_thr for v in var_h)

                is_abn = h_ent >= CONFIRM and h_var >= CONFIRM
                is_wrn = (avg_e > ent_thr * 0.8 and avg_v > var_thr * 0.8 and not is_abn)

                if is_abn:   status = "abnormal"; n_abn += 1
                elif is_wrn or is_rising(list(ent_h)): status = "warning"; n_warn += 1
                else:         status = "normal";   n_norm += 1

                if fidx % 3 == 0:
                    pct_val = 100.0 if is_live else round(fidx / max(total, 1) * 100, 1)
                    push_event("prediction", {
                        "frame": fidx, "seq": fidx // 3,
                        "ent": round(avg_e, 4), "var": round(avg_v, 4),
                        "status": status,
                        "pct": pct_val,
                    })

                if is_abn and cooldown == 0:
                    alerts += 1
                    with state_lock:
                        processing_state["alert_count"] = alerts
                    add_log(f"ALERT frame {fidx} — ent={ent:.3f} var={var:.3f}", "alert")
                    with open("alert.txt", "w") as f:
                        f.write(f"signal=1\nframe={fidx}\n")
                    push_event("alert", {
                        "frame": fidx, "ent": round(ent, 4), "var": round(var, 4),
                        "actions": [
                            "Gate opening command issued",
                            "Security personnel notified",
                            "Crowd management broadcast triggered",
                        ]
                    })
                    push_event("signal",  {"signal": 1, "frame": fidx})
                    push_event("action",  {
                        "frame": fidx,
                        "ent": round(ent, 4),
                        "actions": [
                            "Gate opening command issued",
                            "Security personnel notified",
                            "Crowd management broadcast triggered",
                        ]
                    })
                    add_log("Security Alert Activated", "action")
                    cooldown = 25

                if cooldown > 0: cooldown -= 1
                mag_seq.pop(0); angle_seq.pop(0)

            prev = frame.copy()
            fidx += 1

        cap.release()
        if os.path.exists("alert.txt"): os.remove("alert.txt")

        total_seq = n_norm + n_warn + n_abn
        final = {
            "total_frames": fidx, "total_sequences": total_seq,
            "normal_count": n_norm, "warning_count": n_warn,
            "abnormal_count": n_abn,
            "detection_rate": round(n_abn / max(total_seq,1) * 100, 2),
            "alert_count": alerts,
            "entropy_thresh": ent_thr, "var_thresh": var_thr,
        }
        if current_run_id == run_id:
            with state_lock:
                processing_state["running"] = False
                processing_state["stage"]   = "done"

            add_log(f"Done — {alerts} alerts raised", "success")
            push_event("complete", {"metrics": final})
    except Exception as e:
        if current_run_id == run_id:
            add_log(f"Pipeline crashed: {str(e)}", "error")
            push_event("error", {"message": f"Connection failed. Ensure phone URL is accessible."})
            with state_lock:
                processing_state["running"] = False
                processing_state["stage"]   = "error"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global pipeline_thread
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    file      = request.files["video"]
    ent_thr   = float(request.form.get("ent_thresh", ENTROPY_THRESH))
    var_thr   = float(request.form.get("var_thresh", VAR_THRESH))
    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    
    with state_lock:
        processing_state["stop_requested"] = True
    if pipeline_thread is not None and pipeline_thread.is_alive():
        pipeline_thread.join(timeout=3.0)
    
    with state_lock:
        global current_run_id
        current_run_id = str(uuid.uuid4())
        processing_state.update({
            "running": True, "events": [], "alert_count": 0,
            "log": [], "stage": "starting", "stop_requested": False
        })
        
    pipeline_thread = threading.Thread(target=run_pipeline, args=(save_path, ent_thr, var_thr, current_run_id), daemon=True)
    pipeline_thread.start()
    return jsonify({"status": "started"})

@app.route("/live", methods=["POST"])
def live():
    global pipeline_thread
    camera_url = request.form.get("camera_url")
    ent_thr    = float(request.form.get("ent_thresh", ENTROPY_THRESH))
    var_thr    = float(request.form.get("var_thresh", VAR_THRESH))
    
    if not camera_url:
        return jsonify({"error": "No camera URL provided"}), 400
        
    if camera_url.isdigit():
        camera_url = int(camera_url)
        
    with state_lock:
        processing_state["stop_requested"] = True
    if pipeline_thread is not None and pipeline_thread.is_alive():
        pipeline_thread.join(timeout=3.0)
        
    with state_lock:
        global current_run_id
        current_run_id = str(uuid.uuid4())
        processing_state.update({
            "running": True, "events": [], "alert_count": 0,
            "log": [], "stage": "starting", "stop_requested": False
        })
        
    pipeline_thread = threading.Thread(target=run_pipeline, args=(camera_url, ent_thr, var_thr, current_run_id), daemon=True)
    pipeline_thread.start()
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_pipeline():
    with state_lock:
        processing_state["stop_requested"] = True
    return jsonify({"status": "stopping"})

@app.route("/stream")
def stream():
    def gen():
        last = 0
        while True:
            with state_lock:
                evts    = processing_state["events"]
                new     = evts[last:]
                last    = len(evts)
                running = processing_state["running"]
                stage   = processing_state["stage"]
            for e in new:
                yield f"data: {json.dumps(e)}\n\n"
            if not running and stage in ("done","error") and not new:
                yield f"data: {json.dumps({'type':'end'})}\n\n"
                break
            time.sleep(0.08)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/status")
def status():
    with state_lock:
        return jsonify(processing_state.copy())

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  CROWD INSTABILITY DETECTION")
    print("  Motion Entropy + Directional Variance")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000, threaded=True)
