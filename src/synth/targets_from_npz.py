# src/synth/targets_from_npz.py
import numpy as np

EPS = 1e-8


def ema_smooth(x, alpha=0.2):
    """Simple exponential moving average smoothing."""
    if len(x) == 0:
        return x
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def robust_median_f0(f0, vuv, fmin):
    """Median F0 on voiced frames (> fmin). Fallback to 110 Hz if none."""
    mask = (vuv > 0.5) & (f0 > fmin)
    if mask.sum() == 0:
        return 110.0
    return float(np.median(f0[mask]))


def estimate_t60_from_loudness_db(loud_db, sr, hop):
    """
    Stima T60 (sec) dal decadimento di loudness (dB) dopo il picco.
    Fitta la pendenza su (peak..end) di L0 - L(t). T60 ~ 60 / slope.
    Ritorna in secondi. NON normalizzato.
    """
    if len(loud_db) < 4:
        return 0.3
    ld = ema_smooth(loud_db, alpha=0.2)
    i0 = int(np.argmax(ld))
    L0 = float(ld[i0])
    t = np.arange(len(ld)) * (hop / sr)
    x = t[i0:] - t[i0]
    y = (L0 - ld[i0:])  
    mask = y >= 0.0
    x, y = x[mask], y[mask]
    if len(x) < 3 or y.max() < 1e-3:
        return 0.3
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]  # dB/sec
    if slope <= 1e-6:
        return 0.3
    t60 = 60.0 / slope
    return float(t60)


def spectral_brightness_ratio(mel):
    """Rapporto di energia nell'ultimo terzo delle mel bands."""
    if mel.ndim != 2:
        mel = np.atleast_2d(mel)
    n_mels = mel.shape[0]
    cutoff = int(n_mels * (2.0 / 3.0))
    top = mel[cutoff:, :]
    num = float(top.sum())
    den = float(mel.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))


def noise_mix_from_hn(mel_h, mel_p):
    """Quota di energia percussiva/noise (dalla separazione H/P)."""
    if mel_h is None or mel_p is None:
        return 0.2
    num = float(mel_p.sum())
    den = float(mel_h.sum() + mel_p.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))


def pick_position_from_brightness(bright):
    """
    Bridge -> bright (pos bassa); Neck -> dark (pos alta).
    Mappa brightness [0,1] -> posizione [0.1, 0.9] (invertita).
    """
    return float(np.clip(0.9 - 0.8 * bright, 0.1, 0.9))


def targets_from_npz(
    npz_dict,
    sr,
    hop,
    fmin_hz=30.0,
    fmax_hz=12000.0,
    decay_min=None,          
    decay_max=None,      
):
    """
    Costruisce il vettore target 6D in **unità reali**:
        [pitch_hz, decay_t60(sec), brightness(0..1), damping(0..1), pick_pos(0.1..0.9), noise_mix(0..1)]

    - decay_t60 è in SECONDI.
    - Se decay_min/max sono passati, si fa clamp su T60 e si calcola damping con gli **stessi** bound.
      (Coerenza tra mapping e normalizzazione dei parametri.)
    """
    f0    = npz_dict["f0"].astype(float)
    vuv   = npz_dict["vuv"].astype(float)
    loud  = npz_dict["loud"].astype(float)   # dB
    mel   = npz_dict["mel"].astype(float)    # (n_mels, T)
    mel_h = npz_dict.get("mel_h", None)
    mel_p = npz_dict.get("mel_p", None)

    pitch_hz  = robust_median_f0(f0, vuv, fmin = fmin_hz)
    t60       = estimate_t60_from_loudness_db(loud, sr, hop)   # sec 
    bright    = spectral_brightness_ratio(mel)
    noise_mix = noise_mix_from_hn(mel_h, mel_p)

    # Clamp opzionale di T60 ai bound desiderati
    if decay_min is not None or decay_max is not None:
        lo = decay_min if decay_min is not None else t60
        hi = decay_max if decay_max is not None else t60
        t60 = float(np.clip(t60, lo, hi))

    # Damping coerente con gli stessi bound usati per T60
    if decay_min is not None and decay_max is not None:
        d0, d1 = float(decay_min), float(decay_max)
    else:
        d0, d1 = 0.05, 2.5  
    damping = 1.0 - (t60 - d0) / (d1 - d0)
    damping = float(np.clip(damping, 0.0, 1.0))

    pick_pos = pick_position_from_brightness(bright)

    return np.array(
        [pitch_hz, t60, bright, damping, pick_pos, noise_mix],
        dtype=np.float32,
    )
