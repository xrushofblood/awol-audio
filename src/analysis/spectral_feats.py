# src/analysis/spectral_feats.py
from __future__ import annotations
import numpy as np
import soundfile as sf
from typing import Dict, Tuple, Optional


# -----------------------
# Utilities (numerically safe)
# -----------------------
def _to_mono_f32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y

def _rms(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32)
    return float(np.sqrt(np.mean(y ** 2)) + 1e-12)

def _frame_rms_envelope(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Frame-level RMS (no window overlap handling needed for coarse decay)."""
    y = np.asarray(y, dtype=np.float32)
    nb = max(1, len(y) // hop)
    env = np.empty(nb, dtype=np.float32)
    for i in range(nb):
        seg = y[i * hop : (i + 1) * hop]
        if seg.size == 0:
            env[i] = 0.0
        else:
            env[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return env

# -----------------------
# F0 via autocorrelation (simple/robust)
# -----------------------
def _median_f0_hz(
    y: np.ndarray,
    sr: int,
    fmin: float = 70.0,
    fmax: float = 1200.0,
) -> float:
    """
    Simple ACF pitch estimate on the first analysis window.
    Returns 0.0 if too short, too flat, or no valid lag.
    """
    y = _to_mono_f32(y)
    if len(y) < sr // 20:  # < 50 ms not enough
        return 0.0

    # Analysis window ~ 100 ms (clamped to signal length)
    n = min(len(y), int(0.1 * sr))
    x = y[:n].copy()
    x -= np.mean(x)
    if not np.any(np.abs(x) > 0):
        return 0.0

    # Light pre-emphasis & Hann window
    x[1:] -= 0.97 * x[:-1]
    x *= np.hanning(len(x)).astype(np.float32)

    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]  # keep non-negative lags
    ac[0] = 0.0

    # Valid lag range from fmax..fmin
    lag_min = int(np.floor(sr / max(fmax, 1.0)))
    lag_max = int(np.ceil(sr / max(fmin, 1.0)))
    lag_min = max(lag_min, 1)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_max <= lag_min:
        return 0.0

    # Peak search in the valid window
    window = ac[lag_min : lag_max + 1]
    lag_rel = int(np.argmax(window))
    lag = lag_min + lag_rel
    if lag <= 0:
        return 0.0

    f0 = sr / float(lag)
    # Sanity clamp
    if f0 < fmin or f0 > fmax:
        return 0.0
    return float(f0)

# -----------------------
# Spectral centroid & rolloff85 on power spectrum
# -----------------------
def _spectral_centroid_and_rolloff(y: np.ndarray, sr: int) -> Tuple[float, float]:
    y = _to_mono_f32(y)
    n = 4096 if len(y) >= 4096 else len(y)
    if n <= 8:
        return 0.0, 0.0
    win = np.hanning(n).astype(np.float32)
    x = y[:n] * win
    spec = np.fft.rfft(x)
    mag2 = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    power_sum = np.sum(mag2) + 1e-20
    centroid = float(np.sum(freqs * mag2) / power_sum)

    cumsum = np.cumsum(mag2)
    thresh = 0.85 * cumsum[-1]
    idx = int(np.searchsorted(cumsum, thresh))
    idx = np.clip(idx, 0, len(freqs) - 1)
    roll85 = float(freqs[idx])

    # Clamp to [0, Nyquist] for safety
    nyq = sr / 2.0
    centroid = float(np.clip(centroid, 0.0, nyq))
    roll85 = float(np.clip(roll85, 0.0, nyq))
    return centroid, roll85

# -----------------------
# Decay T20 / T40 from log-envelope tail
# -----------------------
def _decay_times(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Estimate T20/T40 by linear fit on the log-RMS tail.
    Returns 0,0 if dynamic range is insufficient or fit is ill-posed.
    """
    y = _to_mono_f32(y)
    if len(y) < sr // 20:
        return 0.0, 0.0

    hop = max(1, sr // 200)  # ~5 ms
    env = _frame_rms_envelope(y, sr, hop)
    if env.size < 10:
        return 0.0, 0.0

    env_db = 20.0 * np.log10(np.maximum(env, 1e-12))
    # Use the last 50% to approximate the "tail"
    start = env_db.size // 2
    x = np.arange(env_db.size - start, dtype=np.float32) * (hop / sr)
    ydb = env_db[start:]

    # Need roughly decreasing tail (negative slope); otherwise fail
    # Fit ydb = m * t + b
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        m, b = np.linalg.lstsq(A, ydb, rcond=None)[0]
    except Exception:
        return 0.0, 0.0

    if not np.isfinite(m) or m >= -1e-8:
        return 0.0, 0.0

    # T20/T40: time to drop by 20/40 dB given slope m [dB/s]
    T20 = 20.0 / (-m)
    T40 = 40.0 / (-m)

    # Guard rails: unrealistic long values -> zero (means "not measurable")
    if T20 > 10.0:
        T20 = 0.0
    if T40 > 10.0:
        T40 = 0.0

    return float(T20), float(T40)

# -----------------------
# Public API
# -----------------------
def extract_all_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Returns the dict expected by scripts/analyze_generated_audio.py
    Keys:
      - rms
      - f0_hz_med
      - spectral_centroid_hz_med
      - rolloff85_hz_med
      - decay_T20_sec
      - decay_T40_sec
    """
    rms = _rms(y)
    f0 = _median_f0_hz(y, sr)
    sc, roll = _spectral_centroid_and_rolloff(y, sr)
    T20, T40 = _decay_times(y, sr)
    return {
        "rms": rms,
        "f0_hz_med": f0,
        "spectral_centroid_hz_med": sc,
        "rolloff85_hz_med": roll,
        "decay_T20_sec": T20,
        "decay_T40_sec": T40,
    }

def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = _to_mono_f32(y)
    return y, int(sr)