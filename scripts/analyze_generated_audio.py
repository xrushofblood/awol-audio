# scripts/analyze_generated_audio.py
import os, re, csv, glob, numpy as np
import soundfile as sf
import librosa

IN_DIR = "outputs/gen_audio"
OUT_CSV = "outputs/gen_audio/analysis_metrics.csv"

def rms_db(x):
    eps = 1e-12
    return 20*np.log10(np.sqrt(np.mean(x**2))+eps)

def estimate_decay_time(x, sr, floor_db=-40.0):
    """Approximate decay time until the RMS envelope drops below (peak + floor_db)."""
    frame = int(0.010*sr)
    hop = int(0.005*sr)
    if len(x) < frame: 
        return 0.0
    rms_env = librosa.feature.rms(y=x, frame_length=frame, hop_length=hop).flatten()
    peak = np.max(rms_env)
    if peak <= 0:
        return 0.0
    thr = peak * 10**(floor_db/20.0)
    idx = np.where(rms_env < thr)[0]
    if len(idx)==0:
        # never reached threshold within file
        return len(x)/sr
    t = idx[0] * hop / sr
    return t

def estimate_f0(x, sr):
    """Median F0 by librosa.pyin (robust). Returns Hz, or 0 if undefined."""
    f0, _, _ = librosa.pyin(x, fmin=50, fmax=2000, sr=sr, frame_length=2048)
    if f0 is None:
        return 0.0
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size>0 else 0.0

def spectral_centroid(x, sr):
    c = librosa.feature.spectral_centroid(y=x, sr=sr).flatten()
    return float(np.median(c)) if c.size>0 else 0.0

def spectral_rolloff(x, sr, roll_percent=0.85):
    r = librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=roll_percent).flatten()
    return float(np.median(r)) if r.size>0 else 0.0

def parse_prompt_from_filename(fn):
    base = os.path.basename(fn).replace(".wav","")
    p = base.replace("_"," ").replace("-", " ")
    return p

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
rows = []

wavs = sorted(glob.glob(os.path.join(IN_DIR, "*.wav")))
for wav in wavs:
    x, sr = sf.read(wav)
    if x.ndim>1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    # metrics
    f0 = estimate_f0(x, sr)
    cent = spectral_centroid(x, sr)
    roll = spectral_rolloff(x, sr, 0.85)
    decay_t20 = estimate_decay_time(x, sr, floor_db=-20.0)
    decay_t40 = estimate_decay_time(x, sr, floor_db=-40.0)
    rms = np.sqrt(np.mean(x**2))

    rows.append({
        "file": os.path.basename(wav),
        "prompt_guess": parse_prompt_from_filename(wav),
        "sr": sr,
        "rms": f"{rms:.6f}",
        "f0_hz_med": f"{f0:.2f}",
        "spectral_centroid_hz_med": f"{cent:.2f}",
        "rolloff85_hz_med": f"{roll:.2f}",
        "decay_T20_sec": f"{decay_t20:.4f}",
        "decay_T40_sec": f"{decay_t40:.4f}",
        "duration_sec": f"{len(x)/sr:.3f}",
    })

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"[ANALYSIS] Wrote {len(rows)} rows to {OUT_CSV}")
print("Tip: sort by f0_hz_med (pitch), spectral_centroid (brightness), decay_T40_sec (sustain).")
