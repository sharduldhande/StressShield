"""
app.py — StressShield web console.

Streams real-time stress predictions from BITalino via LSL to a browser UI.
Supports per-user calibration to personalise the stress threshold.

Usage:
    pip install flask
    python app.py
    Open http://localhost:5000
"""

import json
import os
import queue
import threading
import time
import warnings
from collections import deque

import joblib
import neurokit2 as nk
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify, send_from_directory
from pylsl import StreamInlet, resolve_streams

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH        = "stress_model_no_eda.pkl"
CALIBRATION_PATH  = "calibration.json"
STREAM_NAME       = "OpenSignals"
FS                = 1000
WINDOW_SEC        = 60
STEP_SEC          = 5
WINDOW            = WINDOW_SEC * FS
CH_RESP           = 1
CH_ECG            = 2
SIGNAL_DOWNSAMPLE = 10
CALIB_VIDEO_FILE  = "4minbeach.mp4"
CALIB_USE_WINDOWS = 24   # last 2 min of 4 min video (24 × 5s = 120s)

# ── global state ──────────────────────────────────────────────────────────────

app = Flask(__name__)
event_queue:  queue.Queue = queue.Queue(maxsize=50)
signal_queue: queue.Queue = queue.Queue(maxsize=200)

buf_ecg  = None
buf_resp = None

active_profile    = None
calibration_state = {"pending": False, "running": False, "name": None, "windows": []}

# ── export state ─────────────────────────────────────────────────────────────

_export: dict = {
    "active":     False,
    "sig_ts":     [],
    "sig_ecg":    [],
    "sig_resp":   [],
    "pred_rows":  [],   # one tuple per 5-s prediction window
    "duration":   0,
    "started_at": None,
}
_export_lock = threading.Lock()

# ── calibration persistence ───────────────────────────────────────────────────

def load_calibrations() -> dict:
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


def save_calibration(name: str, threshold: float, baseline_mean: float, baseline_std: float):
    cals = load_calibrations()
    cals[name] = {
        "threshold":     round(threshold, 4),
        "baseline_mean": round(baseline_mean, 4),
        "baseline_std":  round(baseline_std, 4),
        "created":       time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cals, f, indent=2)


def get_threshold() -> float:
    if active_profile is None:
        return 0.5
    cals = load_calibrations()
    return cals.get(active_profile, {}).get("threshold", 0.5)

# ── feature extraction ────────────────────────────────────────────────────────

def ecg_features(ecg: np.ndarray, fs: int) -> dict:
    try:
        signals, _ = nk.ecg_process(ecg, sampling_rate=fs)
        hrv = nk.hrv(signals, sampling_rate=fs, show=False)
        return {
            "hrv_meanNN": hrv["HRV_MeanNN"].iloc[0],
            "hrv_sdnn":   hrv["HRV_SDNN"].iloc[0],
            "hrv_rmssd":  hrv["HRV_RMSSD"].iloc[0],
            "hrv_pnn50":  hrv["HRV_pNN50"].iloc[0],
            "hrv_lf":     hrv["HRV_LF"].iloc[0],
            "hrv_hf":     hrv["HRV_HF"].iloc[0],
            "hrv_lf_hf":  hrv["HRV_LFHF"].iloc[0],
        }
    except Exception:
        return {k: np.nan for k in [
            "hrv_meanNN", "hrv_sdnn", "hrv_rmssd",
            "hrv_pnn50", "hrv_lf", "hrv_hf", "hrv_lf_hf",
        ]}


def resp_features(resp: np.ndarray, fs: int) -> dict:
    try:
        signals, _ = nk.rsp_process(resp, sampling_rate=fs)
        rate = signals["RSP_Rate"].values
        return {
            "resp_rate_mean":      np.mean(rate),
            "resp_rate_std":       np.std(rate),
            "resp_rate_min":       np.min(rate),
            "resp_rate_max":       np.max(rate),
            "resp_amplitude_mean": np.mean(signals["RSP_Amplitude"].values),
        }
    except Exception:
        return {k: np.nan for k in [
            "resp_rate_mean", "resp_rate_std", "resp_rate_min",
            "resp_rate_max", "resp_amplitude_mean",
        ]}


# ── ring buffer ───────────────────────────────────────────────────────────────

class ChannelBuffer:
    def __init__(self, maxlen: int):
        self._buf  = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def extend(self, samples):
        with self._lock:
            self._buf.extend(samples)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return np.array(self._buf)

    def __len__(self):
        with self._lock:
            return len(self._buf)


# ── event helpers ─────────────────────────────────────────────────────────────

def push(event_type: str, data: dict):
    try:
        event_queue.put_nowait({"type": event_type, "data": data})
    except queue.Full:
        pass


def push_signal(ecg_chunk: list, resp_chunk: list):
    try:
        signal_queue.put_nowait({"type": "signal", "data": {"ecg": ecg_chunk, "resp": resp_chunk}})
    except queue.Full:
        pass


# ── background worker ─────────────────────────────────────────────────────────

def detection_worker():
    global buf_ecg, buf_resp

    try:
        artifact  = joblib.load(MODEL_PATH)
        model     = artifact["model"]
        scaler    = artifact["scaler"]
        feat_cols = artifact["features"]
    except Exception as e:
        push("error", {"message": f"Failed to load model: {e}"})
        return

    push("status", {"message": f"Looking for LSL stream '{STREAM_NAME}'..."})
    all_streams = resolve_streams(wait_time=5.0)
    streams = [s for s in all_streams if STREAM_NAME.lower() in s.name().lower()]
    if not streams:
        push("error", {"message": "LSL stream not found. Is OpenSignals running?"})
        return

    inlet = StreamInlet(streams[0])
    push("status", {"message": f"Connected. Buffering {WINDOW_SEC}s of data..."})

    buf_ecg  = ChannelBuffer(maxlen=WINDOW)
    buf_resp = ChannelBuffer(maxlen=WINDOW)

    stop_event = threading.Event()

    def reader():
        while not stop_event.is_set():
            samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=256)
            if not samples:
                continue
            buf_ecg.extend( [s[CH_ECG]  for s in samples])
            buf_resp.extend([s[CH_RESP] for s in samples])
            ecg_ds  = [round(s[CH_ECG],  4) for i, s in enumerate(samples) if i % SIGNAL_DOWNSAMPLE == 0]
            resp_ds = [round(s[CH_RESP], 4) for i, s in enumerate(samples) if i % SIGNAL_DOWNSAMPLE == 0]
            if ecg_ds:
                push_signal(ecg_ds, resp_ds)
            with _export_lock:
                if _export["active"]:
                    elapsed = time.time() - _export["started_at"]
                    if elapsed >= _export["duration"]:
                        _export["active"] = False
                        push("export_done", {})
                    else:
                        for s in samples:
                            t = time.time() - _export["started_at"]
                            _export["sig_ts"].append(round(t, 6))
                            _export["sig_ecg"].append(round(s[CH_ECG], 6))
                            _export["sig_resp"].append(round(s[CH_RESP], 6))

    threading.Thread(target=reader, daemon=True).start()

    last_predict_time = None

    try:
        while True:
            now = time.time()
            n   = len(buf_ecg)

            if n < WINDOW:
                push("buffering", {"pct": round(n / WINDOW * 100, 1)})
                time.sleep(1.0)
                continue

            if last_predict_time and (now - last_predict_time) < STEP_SEC:
                time.sleep(0.5)
                continue

            last_predict_time = now
            ecg_win  = buf_ecg.snapshot()
            resp_win = buf_resp.snapshot()

            feats = {}
            feats.update(ecg_features(ecg_win,  FS))
            feats.update(resp_features(resp_win, FS))
            x = np.array([feats.get(f, np.nan) for f in feat_cols]).reshape(1, -1)

            if np.isnan(x).any():
                push("prediction", {
                    "ts":       time.strftime("%H:%M:%S"),
                    "step_sec": STEP_SEC,
                    "error":    "Signal quality too low — could not extract features",
                })
                continue

            x_scaled = scaler.transform(x)
            prob = float(model.predict_proba(x_scaled)[0, 1])

            # start pending calibration now that buffer is ready
            if calibration_state["pending"]:
                calibration_state["pending"] = False
                calibration_state["running"] = True
                push("calib_started", {})

            # calibration collection
            if calibration_state["running"]:
                calibration_state["windows"].append(prob)
                n_done = len(calibration_state["windows"])
                push("calib_progress", {"n": n_done})
                continue

            threshold = get_threshold()
            label     = 1 if prob >= threshold else 0

            hr = round(60000 / feats["hrv_meanNN"], 1) if feats["hrv_meanNN"] > 0 else None
            metrics = {
                "hr":        hr,
                "rmssd":     round(feats["hrv_rmssd"],           1),
                "lf_hf":     round(feats["hrv_lf_hf"],           2),
                "resp_rate": round(feats["resp_rate_mean"],      1),
                "resp_amp":  round(feats["resp_amplitude_mean"], 3),
            }

            ev = {
                "ts":        time.strftime("%H:%M:%S"),
                "label":     label,
                "state":     "Stress" if label == 1 else "Non-stress",
                "prob":      round(prob, 3),
                "threshold": round(threshold, 3),
                "step_sec":  STEP_SEC,
                "metrics":   metrics,
            }
            push("prediction", ev)
            with _export_lock:
                if _export["active"]:
                    _export["pred_rows"].append((
                        ev["ts"], ev["prob"], label, ev["state"],
                        metrics.get("hr"), metrics.get("rmssd"),
                        metrics.get("lf_hf"), metrics.get("resp_rate"),
                        metrics.get("resp_amp"),
                    ))

    except Exception as e:
        push("error", {"message": str(e)})
    finally:
        stop_event.set()


# ── SSE endpoints ─────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                event = event_queue.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/signals")
def signals():
    def generate():
        while True:
            try:
                event = signal_queue.get(timeout=5)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── calibration API ───────────────────────────────────────────────────────────

@app.route("/api/profiles")
def api_profiles():
    cals = load_calibrations()
    return jsonify({
        "profiles": [
            {"name": k, "threshold": v["threshold"], "created": v["created"]}
            for k, v in cals.items()
        ],
        "active": active_profile,
    })


@app.route("/api/select", methods=["POST"])
def api_select():
    global active_profile
    active_profile = request.json.get("name")
    return jsonify({"ok": True, "active": active_profile, "threshold": get_threshold()})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    global calibration_state
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    if calibration_state["running"] or calibration_state["pending"]:
        return jsonify({"error": "Calibration already running"}), 400
    already_buffered = buf_ecg is not None and len(buf_ecg) >= WINDOW
    calibration_state = {
        "pending": not already_buffered,
        "running": already_buffered,
        "name":    name,
        "windows": [],
    }
    if already_buffered:
        push("calib_started", {})
    return jsonify({"ok": True, "name": name})


@app.route("/api/calibrate/finish", methods=["POST"])
def api_calibrate_finish():
    global calibration_state
    if not calibration_state["running"] and not calibration_state["pending"]:
        return jsonify({"error": "No calibration running"}), 400
    windows = calibration_state["windows"]
    name = (request.json or {}).get("name", "unknown")
    calibration_state = {"pending": False, "running": False, "name": None, "windows": []}
    if len(windows) < 2:
        return jsonify({"error": "Not enough data collected"}), 400
    probs_arr     = np.array(windows[-CALIB_USE_WINDOWS:])
    baseline_mean = float(np.mean(probs_arr))
    baseline_std  = float(np.std(probs_arr))
    threshold     = min(baseline_mean + 1.5 * baseline_std, 0.95)
    save_calibration(name, threshold, baseline_mean, baseline_std)
    push("calib_done", {
        "name":      name,
        "threshold": round(threshold, 3),
        "mean":      round(baseline_mean, 3),
    })
    return jsonify({"ok": True})


@app.route("/media/<path:filename>")
def serve_media(filename):
    return send_from_directory(".", filename)


@app.route("/api/profiles/<name>", methods=["DELETE"])
def api_delete_profile(name):
    global active_profile
    cals = load_calibrations()
    if name not in cals:
        return jsonify({"error": "Profile not found"}), 404
    del cals[name]
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cals, f, indent=2)
    if active_profile == name:
        active_profile = None
    return jsonify({"ok": True})


# ── export routes ────────────────────────────────────────────────────────────

@app.route("/api/export/start", methods=["POST"])
def api_export_start():
    import io, csv  # noqa: F401 — csv imported here; also used in download routes
    seconds = int(request.json.get("seconds", 30))
    seconds = max(1, min(seconds, 300))
    with _export_lock:
        _export.update(
            active=True,
            sig_ts=[], sig_ecg=[], sig_resp=[],
            pred_rows=[],
            duration=seconds,
            started_at=time.time(),
        )
    return jsonify(ok=True, seconds=seconds)


@app.route("/api/export/signals")
def api_export_signals():
    import io, csv
    with _export_lock:
        if _export["active"] or not _export["sig_ts"]:
            return jsonify(error="No export ready"), 404
        rows = list(zip(_export["sig_ts"], _export["sig_ecg"], _export["sig_resp"]))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time_s", "ecg", "resp"])
    w.writerows(rows)
    buf.seek(0)
    return Response(
        buf,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="signals_export.csv"'},
    )


@app.route("/api/export/predictions")
def api_export_predictions():
    import io, csv
    with _export_lock:
        if _export["active"] or not _export["pred_rows"]:
            return jsonify(error="No export ready"), 404
        rows = list(_export["pred_rows"])
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["time", "prob", "label", "state",
                "hr_bpm", "rmssd_ms", "lf_hf", "resp_rate_brpm", "resp_amp"])
    w.writerows(rows)
    buf.seek(0)
    return Response(
        buf,
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions_export.csv"'},
    )


# ── main page ─────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StressShield</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  /* ── colour tokens ── */
  body {
    --bg:        #0f1117;
    --bg-card:   #1a1d27;
    --bg-input:  #12141c;
    --bg-btn:    #1e2130;
    --border:    #2a2d3a;
    --text:      #e0e0e0;
    --text-muted:#aaa;
    --text-dim:  #555;
    --text-card: #3a3d4a;
    --divider:   #222530;
  }
  body.light {
    --bg:        #f0f2f7;
    --bg-card:   #ffffff;
    --bg-input:  #e8eaf0;
    --bg-btn:    #e2e5ee;
    --border:    #c8ccd8;
    --text:      #1a1d27;
    --text-muted:#444;
    --text-dim:  #888;
    --text-card: #9098b0;
    --divider:   #d8dce8;
  }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 20px;
    gap: 10px;
    transition: background 0.2s, color 0.2s;
  }

  /* ── top bar ── */
  #topbar {
    width: 100%;
    max-width: 1100px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }

  #topbar h1 {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-right: auto;
  }

  #profile-select {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 0.78rem;
    min-width: 180px;
  }

  .btn {
    background: var(--bg-btn);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 7px;
    padding: 5px 13px;
    font-size: 0.78rem;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
  }
  .btn:hover         { background: var(--border); color: var(--text); }
  .btn-primary       { background: #1e3a6e; border-color: #2a5aaa; color: #7ab0ff; }
  .btn-primary:hover { background: #2a5aaa; color: #fff; }
  .btn-danger        { background: #3a1a1a; border-color: #6a2a2a; color: #ff9090; }
  .btn-danger:hover  { background: #6a2a2a; color: #fff; }

  /* ── calibration panel ── */
  #calib-panel {
    width: 100%;
    max-width: 1100px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    display: none;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  #calib-panel.open { display: flex; }

  #calib-name {
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 0.78rem;
    flex: 1;
    min-width: 140px;
  }
  #calib-status { font-size: 0.72rem; color: var(--text-dim); flex: 1; }

  /* ── layout ── */
  .layout {
    width: 100%;
    max-width: 1100px;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .top-row {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 200px;
    align-items: stretch;
    gap: 10px;
  }

  .bottom-row {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    align-items: stretch;
    gap: 10px;
  }

  /* ── cards ── */
  .card {
    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    transition: border-color 0.4s, background 0.2s;
    overflow: hidden;
    min-height: 0;
  }
  .card-stress { border-color: #ff5c5c55 !important; }
  .card-calm   { border-color: #4cff9155 !important; }

  .card-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-card);
    margin-bottom: 8px;
  }

  /* ── trend chart ── */
  #trend-card {
    padding: 12px 14px 8px;
    display: flex;
    flex-direction: column;
  }
  #trend-card canvas { flex: 1; min-height: 0; }

  #buffer-wrap { margin-top: 10px; }
  #buffer-bar-track {
    background: var(--border); border-radius: 4px;
    height: 4px; overflow: hidden; margin-bottom: 5px;
  }
  #buffer-bar-fill {
    height: 100%; width: 0%; border-radius: 4px;
    background: #4a9eff; transition: width 0.4s;
  }
  #buffer-label { font-size: 0.7rem; color: var(--text-dim); }

  /* ── status panel ── */
  #status-panel {
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 10px;
  }

  #state-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: var(--text-dim);
    transition: color 0.4s;
    line-height: 1.1;
  }
  .stress { color: #ff5c5c !important; }
  .calm   { color: #4cff91 !important; }

  #prob-bar-track {
    background: var(--border); border-radius: 5px; height: 7px; overflow: hidden;
  }
  #prob-bar-fill {
    height: 100%; width: 0%; border-radius: 5px;
    background: #4a9eff; transition: width 0.5s ease, background 0.4s;
  }
  .bar-stress { background: #ff5c5c !important; }
  .bar-calm   { background: #4cff91 !important; }

  .status-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: var(--text-dim);
    font-variant-numeric: tabular-nums;
  }

  .divider { border: none; border-top: 1px solid var(--divider); }

  .stat-item { display: flex; flex-direction: column; gap: 1px; }
  .stat-lbl  { font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-card); }
  .stat-val  { font-size: 0.82rem; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; }

  /* ── signal cards ── */
  .signal-card {
    padding: 10px 12px 8px;
    display: flex;
    flex-direction: column;
  }

  .signal-row {
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: stretch;
    gap: 12px;
  }

  .signal-wrap        { flex: 1; min-width: 0; min-height: 0; }
  .signal-wrap canvas { display: block; width: 100% !important; height: 100% !important; }

  .metrics-col {
    width: 84px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    border-left: 1px solid transparent;
    padding-left: 10px;
    gap: 8px;
  }

  .metric       { display: flex; flex-direction: column; }
  .metric-lbl   { font-size: 0.57rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--text-card); }
  .metric-val   { font-size: 1rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text-muted); }
  .metric-unit  { font-size: 0.58rem; color: var(--text-dim); font-weight: 400; }

  /* ── calib video overlay ── */
  #calib-overlay {
    display: none; position: fixed; inset: 0;
    background: #000; z-index: 100;
    flex-direction: column; align-items: center; justify-content: center; gap: 14px;
  }
  #calib-overlay.active { display: flex; }
  #calib-video { width: 100%; max-width: 900px; border-radius: 8px; }
  #calib-overlay-bar-track {
    width: 100%; max-width: 900px;
    background: #1a1a1a; border-radius: 4px; height: 4px; overflow: hidden;
  }
  #calib-overlay-bar-fill {
    height: 100%; width: 0%;
    background: #4a9eff; border-radius: 4px; transition: width 0.3s;
  }
  #calib-overlay-status { font-size: 0.75rem; color: var(--text-dim); }

  #msg-box { font-size: 0.75rem; color: var(--text-dim); text-align: center; height: 14px; flex-shrink: 0; }
  .msg-error { color: #ff5c5c !important; }
</style>
</head>
<body>

<!-- top bar -->
<div id="topbar">
  <h1>Stress Shield</h1>
  <select id="profile-select">
    <option value="">— No calibration —</option>
  </select>
  <button class="btn btn-danger" id="btn-delete" onclick="deleteProfile()" style="display:none">Delete</button>
  <button class="btn btn-primary" onclick="openCalib()">+ Calibrate</button>
  <button class="btn" id="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark mode">☀️</button>
</div>

<!-- calibration panel -->
<div id="calib-panel">
  <input id="calib-name" type="text" placeholder="Your name…" />
  <button class="btn btn-primary" onclick="startCalib()">Start</button>
  <button class="btn" onclick="closeCalib()">Cancel</button>
  <span id="calib-status"></span>
</div>

<div class="layout">

  <!-- top row -->
  <div class="top-row">

    <div class="card" id="trend-card">
      <div class="card-label">Stress Probability — trend</div>
      <canvas id="trend-chart"></canvas>
      <div id="buffer-wrap" style="display:none">
        <div id="buffer-bar-track"><div id="buffer-bar-fill"></div></div>
        <div id="buffer-label"></div>
      </div>
    </div>

    <div class="card" id="status-panel">
      <div class="card-label">Current state</div>
      <div id="state-label">—</div>
      <div id="prob-bar-track"><div id="prob-bar-fill"></div></div>
      <div class="status-row">
        <span id="prob-label">—</span>
        <span id="threshold-label">—</span>
      </div>
      <hr class="divider">
      <div class="stat-item">
        <span class="stat-lbl">Last updated</span>
        <span class="stat-val" id="ts-label">—</span>
      </div>
      <div class="stat-item">
        <span class="stat-lbl">Next update</span>
        <span class="stat-val" id="countdown">—</span>
      </div>
      <hr class="divider">
      <div class="stat-item">
        <span class="stat-lbl">Export recording</span>
        <div style="display:flex; gap:6px; align-items:center; margin-top:4px;">
          <input id="export-secs" type="number" min="1" max="300" value="30"
                 style="width:52px; background:var(--bg-input); border:1px solid var(--border);
                        color:var(--text-muted); border-radius:6px; padding:3px 6px;
                        font-size:0.75rem;" />
          <span style="font-size:0.7rem; color:var(--text-dim);">sec</span>
          <button class="btn" id="btn-export" onclick="startExport()"
                  style="padding:3px 10px; font-size:0.75rem;">⏺ Record</button>
        </div>
        <span id="export-status" style="font-size:0.68rem; color:var(--text-dim); margin-top:3px;"></span>
      </div>
    </div>

  </div>

  <!-- bottom row -->
  <div class="bottom-row">

    <div class="card signal-card">
      <div class="card-label">ECG</div>
      <div class="signal-row">
        <div class="signal-wrap">
          <canvas id="ecg-chart"></canvas>
        </div>
        <div class="metrics-col">
          <div class="metric">
            <span class="metric-lbl">Heart Rate</span>
            <span class="metric-val" id="m-hr">—<span class="metric-unit"> bpm</span></span>
          </div>
          <div class="metric">
            <span class="metric-lbl">RMSSD</span>
            <span class="metric-val" id="m-rmssd">—<span class="metric-unit"> ms</span></span>
          </div>
          <div class="metric">
            <span class="metric-lbl">LF/HF</span>
            <span class="metric-val" id="m-lfhf">—</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card signal-card">
      <div class="card-label">Respiration</div>
      <div class="signal-row">
        <div class="signal-wrap">
          <canvas id="resp-chart"></canvas>
        </div>
        <div class="metrics-col">
          <div class="metric">
            <span class="metric-lbl">Breath Rate</span>
            <span class="metric-val" id="m-resp-rate">—<span class="metric-unit"> brpm</span></span>
          </div>
          <div class="metric">
            <span class="metric-lbl">Amplitude</span>
            <span class="metric-val" id="m-resp-amp">—</span>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<div id="msg-box"></div>

<!-- calibration video overlay -->
<div id="calib-overlay">
  <video id="calib-video" muted playsinline></video>
  <div id="calib-overlay-bar-track"><div id="calib-overlay-bar-fill"></div></div>
  <div id="calib-overlay-status">Watch the scene and relax — baseline recording in progress</div>
</div>

<script>
// ── trend chart ───────────────────────────────────────────────────────────────

const TREND_MAX  = 40;
const trendTimes = [];
const trendProbs = [];
let   trendThresh = 0.5;

const trendCtx = document.getElementById("trend-chart").getContext("2d");
const trendChart = new Chart(trendCtx, {
  type: "line",
  data: {
    labels: trendTimes,
    datasets: [{
      data: trendProbs,
      borderColor: "#4a9eff",
      backgroundColor: "rgba(74,158,255,0.07)",
      fill: true,
      tension: 0.35,
      pointRadius: 3,
      pointBackgroundColor: [],
      borderWidth: 2,
    }]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      annotation: {
        annotations: {
          threshold: {
            type: "line",
            yMin: trendThresh, yMax: trendThresh,
            borderColor: "#555",
            borderDash: [6, 4],
            borderWidth: 1,
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: "#444", font: { size: 9 }, maxTicksLimit: 6, maxRotation: 0 },
        grid:  { color: "#181b24" },
      },
      y: {
        min: 0, max: 1,
        ticks: { color: "#444", font: { size: 9 }, stepSize: 0.25,
                 callback: v => Math.round(v * 100) + "%" },
        grid: { color: "#181b24" },
      }
    }
  }
});

function updateTrendThreshold(t) {
  trendThresh = t;
  trendChart.options.plugins.annotation.annotations.threshold.yMin = t;
  trendChart.options.plugins.annotation.annotations.threshold.yMax = t;
  trendChart.update("none");
}

function addTrendPoint(ts, prob, isStress) {
  trendTimes.push(ts);
  trendProbs.push(prob);
  if (trendTimes.length > TREND_MAX) { trendTimes.shift(); trendProbs.shift(); }
  const colours = trendProbs.map((_, i) =>
    i === trendProbs.length - 1
      ? (isStress ? "#ff5c5c" : "#4cff91")
      : (trendChart.data.datasets[0].pointBackgroundColor[i] || "#4a9eff")
  );
  trendChart.data.datasets[0].pointBackgroundColor = colours;
  trendChart.data.datasets[0].data   = [...trendProbs];
  trendChart.data.labels             = [...trendTimes];
  trendChart.update("none");
}

// ── signal charts ─────────────────────────────────────────────────────────────

function makeSignalChart(id, color, maxPoints) {
  const ctx  = document.getElementById(id).getContext("2d");
  const data = Array(maxPoints).fill(null);
  return {
    chart: new Chart(ctx, {
      type: "line",
      data: {
        labels: Array(maxPoints).fill(""),
        datasets: [{ data, borderColor: color, borderWidth: 1.5,
                     pointRadius: 0, tension: 0, fill: false }]
      },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, annotation: { annotations: {} } },
        scales: { x: { display: false }, y: { display: false } },
      }
    }),
    data, maxPoints,
  };
}

const ecgChart  = makeSignalChart("ecg-chart",  "#4a9eff", 400);
const respChart = makeSignalChart("resp-chart", "#a78bfa", 1000);

function pushPoints(chartObj, newPoints) {
  chartObj.data.push(...newPoints);
  if (chartObj.data.length > chartObj.maxPoints)
    chartObj.data.splice(0, chartObj.data.length - chartObj.maxPoints);
  chartObj.chart.data.datasets[0].data = chartObj.data;
  chartObj.chart.update("none");
}

const sigEs = new EventSource("/signals");
sigEs.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type !== "signal") return;
  pushPoints(ecgChart,  msg.data.ecg);
  pushPoints(respChart, msg.data.resp);
};

// ── countdown ─────────────────────────────────────────────────────────────────

let nextPredictAt = null;
setInterval(() => {
  if (!nextPredictAt) return;
  const rem = Math.max(0, Math.ceil((nextPredictAt - Date.now()) / 1000));
  document.getElementById("countdown").textContent = rem > 0 ? `${rem}s` : "now";
}, 500);

// ── profile management ────────────────────────────────────────────────────────

async function loadProfiles() {
  const res  = await fetch("/api/profiles");
  const data = await res.json();
  const sel  = document.getElementById("profile-select");
  sel.innerHTML = '<option value="">— No calibration —</option>';
  data.profiles.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = `${p.name}  (thr ${p.threshold})`;
    if (p.name === data.active) opt.selected = true;
    sel.appendChild(opt);
  });
  const active = data.profiles.find(p => p.name === data.active);
  if (active) {
    updateTrendThreshold(active.threshold);
    document.getElementById("threshold-label").textContent = `thr ${Math.round(active.threshold * 100)}%`;
  }
  document.getElementById("btn-delete").style.display = data.active ? "inline-block" : "none";
}

document.getElementById("profile-select").addEventListener("change", async (e) => {
  const name = e.target.value || null;
  const res  = await fetch("/api/select", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name})
  });
  const data = await res.json();
  updateTrendThreshold(data.threshold);
  document.getElementById("threshold-label").textContent = `thr ${Math.round(data.threshold * 100)}%`;
  document.getElementById("btn-delete").style.display = name ? "inline-block" : "none";
});

async function deleteProfile() {
  const name = document.getElementById("profile-select").value;
  if (!name || !confirm(`Delete profile "${name}"?`)) return;
  await fetch(`/api/profiles/${encodeURIComponent(name)}`, {method: "DELETE"});
  await loadProfiles();
}

// ── calibration UI ────────────────────────────────────────────────────────────

function openCalib() {
  document.getElementById("calib-panel").classList.add("open");
  document.getElementById("calib-name").focus();
}

let calibName = null;
const calibVideo       = document.getElementById("calib-video");
const calibOverlay     = document.getElementById("calib-overlay");
const calibOverlayBar  = document.getElementById("calib-overlay-bar-fill");
const calibOverlayStat = document.getElementById("calib-overlay-status");

calibVideo.addEventListener("ended", finishCalib);
calibVideo.addEventListener("timeupdate", () => {
  if (!calibVideo.duration) return;
  calibOverlayBar.style.width = (calibVideo.currentTime / calibVideo.duration * 100) + "%";
});

async function startCalib() {
  const name = document.getElementById("calib-name").value.trim();
  if (!name) { alert("Please enter a name."); return; }
  const res  = await fetch("/api/calibrate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name})
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  calibName = name;
  calibVideo.src = "/media/4minbeach.mp4";
  calibOverlayBar.style.width = "0%";
  calibOverlayStat.textContent = "Watch the scene and relax — baseline recording in progress";
  calibOverlay.classList.add("active");
  calibVideo.play().catch(() => {});
}

async function finishCalib() {
  if (!calibName) return;
  calibOverlay.classList.remove("active");
  calibVideo.pause(); calibVideo.src = "";
  await fetch("/api/calibrate/finish", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({name: calibName})
  });
  calibName = null;
}

function closeCalib() {
  calibOverlay.classList.remove("active");
  calibVideo.pause(); calibVideo.src = "";
  calibName = null;
  document.getElementById("calib-panel").classList.remove("open");
  document.getElementById("calib-name").value = "";
}

// ── SSE event handling ────────────────────────────────────────────────────────

const stateEl    = document.getElementById("state-label");
const probLabel  = document.getElementById("prob-label");
const thrLabel   = document.getElementById("threshold-label");
const barFill    = document.getElementById("prob-bar-fill");
const tsEl       = document.getElementById("ts-label");
const statusCard = document.getElementById("status-panel");
const bufWrap    = document.getElementById("buffer-wrap");
const bufFill    = document.getElementById("buffer-bar-fill");
const bufLabel   = document.getElementById("buffer-label");
const msgBox     = document.getElementById("msg-box");

const es = new EventSource("/stream");

es.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "ping") return;

  if (msg.type === "export_done") {
    const btn    = document.getElementById('btn-export');
    const status = document.getElementById('export-status');
    btn.disabled = false;
    btn.textContent = '\u23fa Record';
    status.textContent = 'Downloading\u2026';
    window.location.href = '/api/export/signals';
    setTimeout(() => { window.open('/api/export/predictions', '_blank'); }, 800);
    setTimeout(() => { status.textContent = 'Done \u2713'; }, 1500);
    return;
  }

  if (msg.type === "status") {
    msgBox.textContent = msg.data.message; msgBox.className = ""; return;
  }
  if (msg.type === "error") {
    msgBox.textContent = msg.data.message; msgBox.className = "msg-error"; return;
  }
  if (msg.type === "buffering") {
    bufWrap.style.display = "block";
    bufFill.style.width   = msg.data.pct + "%";
    bufLabel.textContent  = `Buffering… ${msg.data.pct}%`;
    msgBox.textContent    = "";
    return;
  }
  if (msg.type === "calib_started") {
    document.getElementById("calib-status").textContent = "Recording…";
    return;
  }
  if (msg.type === "calib_progress") {
    const n = msg.data.n;
    calibOverlayStat.textContent =
      (n < 24 ? "Settling in…" : "Capturing baseline…") + `  (${n} windows)`;
    return;
  }
  if (msg.type === "calib_done") {
    document.getElementById("calib-status").textContent =
      `Done — threshold set to ${msg.data.threshold}`;
    setTimeout(async () => {
      await loadProfiles();
      document.getElementById("profile-select").value = msg.data.name;
      document.getElementById("profile-select").dispatchEvent(new Event("change"));
      closeCalib();
    }, 2000);
    return;
  }

  if (msg.type === "prediction") {
    bufWrap.style.display = "none";
    msgBox.textContent    = "";
    if (msg.data.step_sec) nextPredictAt = Date.now() + msg.data.step_sec * 1000;

    const d = msg.data;
    if (d.error) {
      stateEl.textContent = "—"; stateEl.className = "";
      probLabel.textContent = d.error; tsEl.textContent = d.ts;
      statusCard.className = "card"; return;
    }

    const isStress = d.label === 1;
    const pct      = Math.round(d.prob * 100);
    const thrPct   = Math.round(d.threshold * 100);

    stateEl.textContent   = d.state.toUpperCase();
    stateEl.className     = isStress ? "stress" : "calm";
    probLabel.textContent = `${pct}%`;
    thrLabel.textContent  = `thr ${thrPct}%`;
    barFill.style.width   = pct + "%";
    barFill.className     = isStress ? "bar-stress" : "bar-calm";
    tsEl.textContent      = d.ts;
    statusCard.className  = "card " + (isStress ? "card-stress" : "card-calm");

    addTrendPoint(d.ts, d.prob, isStress);

    if (d.metrics) {
      const m = d.metrics;
      document.getElementById("m-hr").innerHTML        = `${m.hr ?? "—"}<span class="metric-unit"> bpm</span>`;
      document.getElementById("m-rmssd").innerHTML     = `${m.rmssd ?? "—"}<span class="metric-unit"> ms</span>`;
      document.getElementById("m-lfhf").textContent    = m.lf_hf ?? "—";
      document.getElementById("m-resp-rate").innerHTML = `${m.resp_rate ?? "—"}<span class="metric-unit"> brpm</span>`;
      document.getElementById("m-resp-amp").textContent = m.resp_amp ?? "—";
    }
  }
};

es.onerror = () => {
  msgBox.textContent = "Connection lost — retrying…"; msgBox.className = "msg-error";
};

async function startExport() {
  const secs   = parseInt(document.getElementById('export-secs').value) || 30;
  const btn    = document.getElementById('btn-export');
  const status = document.getElementById('export-status');
  btn.disabled = true;
  status.textContent = 'Starting\u2026';
  const res = await fetch('/api/export/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({seconds: secs}),
  });
  if (!res.ok) {
    btn.disabled = false;
    status.textContent = 'Error starting export.';
    return;
  }
  let remaining = secs;
  btn.textContent = '\u23f9';
  status.textContent = `Recording\u2026 ${remaining}s left`;
  const iv = setInterval(() => {
    remaining--;
    status.textContent = remaining > 0 ? `Recording\u2026 ${remaining}s left` : 'Finishing\u2026';
    if (remaining <= 0) clearInterval(iv);
  }, 1000);
}

function toggleTheme() {
  const isLight = document.body.classList.toggle('light');
  document.getElementById('theme-toggle').textContent = isLight ? '🌙' : '☀️';
  localStorage.setItem('ss-theme', isLight ? 'light' : 'dark');
}
(function() {
  if (localStorage.getItem('ss-theme') === 'light') {
    document.body.classList.add('light');
    document.getElementById('theme-toggle').textContent = '🌙';
  }
})();

loadProfiles();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=detection_worker, daemon=True).start()
    print("StressShield running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
