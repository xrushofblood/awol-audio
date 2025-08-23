# src/analysis/spectral_feats.py
import numpy as np

def spectral_feats(wave: np.ndarray, sr: int, eps: float = 1e-10):
    """
    Robust features for plucked-like sounds:
      - f0_hz_med    : coarse F0 via autocorrelation peak
      - spectral_centroid_hz_med
      - rolloff85_hz_med
      - decay_T20_sec / decay_T40_sec: linear fit on dB envelope tail
    """
    # --- safety & pre-emphasis ---
    w = wave.astype(np.float32)
    w = np.clip(w, -1.0, 1.0)
    if w.size < 2:
        return dict(f0_hz_med=np.nan, spectral_centroid_hz_med=0.0,
                    rolloff85_hz_med=0.0, decay_T20_sec=np.nan, decay_T40_sec=np.nan)

    w = np.append(w[0], w[1:] - 0.97 * w[:-1])  # pre-emphasis

    # --- window to at least 2048 ---
    N = len(w)
    if N < 2048:
        w = np.pad(w, (0, 2048 - N))
        N = len(w)
    win = np.hanning(N)
    x = w * win

    # --- FFT magnitude ---
    X = np.fft.rfft(x)
    mag = np.abs(X) + eps
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)

    # --- spectral centroid & rolloff85 ---
    centroid_hz = float((freqs * mag).sum() / mag.sum())
    cumsum = np.cumsum(mag)
    thresh = 0.85 * cumsum[-1]
    idx = int(np.searchsorted(cumsum, thresh))
    rolloff85_hz = float(freqs[min(idx, len(freqs) - 1)])

    # --- coarse F0 via autocorrelation (limit to ~60–2000 Hz) ---
    min_lag = max(1, int(sr / 2000))
    max_lag = int(sr / 60)
    ac = np.correlate(x, x, mode='full')[N-1:N-1+max_lag+1]
    ac[:min_lag] = 0
    lag = int(np.argmax(ac))
    f0_hz = float(sr / max(lag, 1))

    # --- decay T20/T40 by linear fit on dB of the envelope tail ---
    env = np.abs(wave).astype(np.float32) + eps
    M = len(env)
    start = int(0.2 * M)  # use the last 80% to avoid attack
    tail = env[start:]
    tail /= tail.max() + eps
    db = 20.0 * np.log10(tail + eps)
    t = np.arange(len(db)) / sr
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, db, rcond=None)[0]

    def tx(dbdrop):
        return (dbdrop - intercept) / slope if slope != 0 else np.nan

    T20 = float(abs(tx(-20.0)))
    T40 = float(abs(tx(-40.0)))

    return dict(
        f0_hz_med=f0_hz,
        spectral_centroid_hz_med=centroid_hz,
        rolloff85_hz_med=rolloff85_hz,
        decay_T20_sec=T20,
        decay_T40_sec=T40
    )
