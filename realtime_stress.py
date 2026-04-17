"""
realtime_stress.py — real-time stress detection from BITalino via LSL.

Connects to the OpenSignals LSL stream, buffers 60s of ECG + respiration,
extracts features every 30s, and runs the trained XGBoost model.

Usage:
    python realtime_stress.py
"""

import time
import threading
import warnings
from collections import deque

import numpy as np
import joblib
import neurokit2 as nk
from pylsl import StreamInlet, resolve_streams

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = "stress_model_no_eda.pkl"
STREAM_NAME  = "OpenSignals"

FS           = 1000          # BITalino sampling rate (Hz)
WINDOW_SEC   = 60
STEP_SEC     = 30
WINDOW       = WINDOW_SEC * FS
STEP         = STEP_SEC   * FS

CH_RESP      = 1             # RESPBIT0
CH_ECG       = 2             # ECGBIT1

# ── feature extraction (mirrors training code) ────────────────────────────────

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
    """Thread-safe fixed-length ring buffer for one channel."""
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


# ── LSL reader thread ─────────────────────────────────────────────────────────

def lsl_reader(inlet: StreamInlet,
               buf_ecg: ChannelBuffer,
               buf_resp: ChannelBuffer,
               stop_event: threading.Event):
    while not stop_event.is_set():
        # pull up to 256 samples at a time
        samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=256)
        if samples:
            buf_ecg.extend( [s[CH_ECG]  for s in samples])
            buf_resp.extend([s[CH_RESP] for s in samples])


# ── inference ─────────────────────────────────────────────────────────────────

def predict(model, scaler, feat_cols,
            ecg_win: np.ndarray, resp_win: np.ndarray) -> tuple[float, int]:
    feats = {}
    feats.update(ecg_features(ecg_win,  FS))
    feats.update(resp_features(resp_win, FS))

    x = np.array([feats.get(f, np.nan) for f in feat_cols]).reshape(1, -1)

    if np.isnan(x).any():
        return None, None

    x_scaled = scaler.transform(x)
    prob  = model.predict_proba(x_scaled)[0, 1]
    label = int(model.predict(x_scaled)[0])
    return prob, label


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # load model
    print(f"Loading model from {MODEL_PATH}...")
    artifact  = joblib.load(MODEL_PATH)
    model     = artifact["model"]
    scaler    = artifact["scaler"]
    feat_cols = artifact["features"]
    print(f"Model ready. Features: {feat_cols}\n")

    # connect to LSL
    print(f"Looking for LSL stream '{STREAM_NAME}'...")
    all_streams = resolve_streams(wait_time=5.0)
    streams = [s for s in all_streams if STREAM_NAME.lower() in s.name().lower()]
    if not streams:
        print("Stream not found. Is OpenSignals running with LSL enabled?")
        return
    inlet = StreamInlet(streams[0])
    print(f"Connected. Buffering {WINDOW_SEC}s before first prediction...\n")

    # buffers
    buf_ecg  = ChannelBuffer(maxlen=WINDOW)
    buf_resp = ChannelBuffer(maxlen=WINDOW)

    # start reader thread
    stop_event = threading.Event()
    reader = threading.Thread(
        target=lsl_reader,
        args=(inlet, buf_ecg, buf_resp, stop_event),
        daemon=True,
    )
    reader.start()

    last_predict_time = None

    try:
        while True:
            now = time.time()

            # wait until buffer is full for the first prediction
            if len(buf_ecg) < WINDOW:
                filled_pct = len(buf_ecg) / WINDOW * 100
                print(f"\r  Buffering... {filled_pct:5.1f}%", end="", flush=True)
                time.sleep(0.5)
                continue

            # after that, predict every STEP_SEC
            if last_predict_time and (now - last_predict_time) < STEP_SEC:
                time.sleep(0.5)
                continue

            last_predict_time = now
            ecg_win  = buf_ecg.snapshot()
            resp_win = buf_resp.snapshot()

            prob, label = predict(model, scaler, feat_cols, ecg_win, resp_win)

            ts = time.strftime("%H:%M:%S")
            if label is None:
                print(f"\n[{ts}]  Feature extraction failed (signal quality issue)")
            else:
                state = "STRESS" if label == 1 else "Non-stress"
                bar   = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                print(f"\n[{ts}]  {state:<12s}  p={prob:.2f}  [{bar}]")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
