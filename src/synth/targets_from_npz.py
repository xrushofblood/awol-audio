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

def robust_median_f0(f0, vuv, fmin=20.0):
    """Median F0 on voiced frames (> fmin). Fallback to 110 Hz if none."""
    mask = (vuv > 0.5) & (f0 > fmin)
    if mask.sum() == 0:
        return 110.0
    return float(np.median(f0[mask]))

def estimate_t60_from_loudness_db(loud_db, sr, hop):
    """
    Estimate T60 (sec) from loudness (dB) decay after peak.
    Fit a slope on (peak..end) of L0 - L(t). T60 ~ 60/slope.
    """
    if len(loud_db) < 4:
        return 0.3
    ld = ema_smooth(loud_db, alpha=0.2)
    i0 = int(np.argmax(ld))
    L0 = float(ld[i0])
    t = np.arange(len(ld)) * (hop / sr)
    x = t[i0:] - t[i0]
    y = (L0 - ld[i0:])  # positive as it decays
    mask = y >= 0.0
    x, y = x[mask], y[mask]
    if len(x) < 3 or y.max() < 1e-3:
        return 0.3
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]  # dB per second
    if slope <= 1e-6:
        return 0.3
    t60 = 60.0 / slope
    return float(np.clip(t60, 0.05, 2.5))

def spectral_brightness_ratio(mel):
    """Ratio of energy in the top 1/3 mel bands (0..1)."""
    if mel.ndim != 2:
        mel = np.atleast_2d(mel)
    n_mels = mel.shape[0]
    cutoff = int(n_mels * (2.0 / 3.0))
    top = mel[cutoff:, :]
    num = float(top.sum())
    den = float(mel.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))

def spectral_brightness_ratio_slice(mel, start, end):
    """Brightness ratio on a mel time slice [start:end]."""
    start = int(max(0, start))
    end = int(min(mel.shape[1], max(start + 1, end)))
    mel_slice = mel[:, start:end]
    return spectral_brightness_ratio(mel_slice)

def noise_mix_from_hn(mel_h, mel_p):
    """Proportion of percussive/noise energy."""
    if mel_h is None or mel_p is None:
        return 0.2
    num = float(mel_p.sum())
    den = float(mel_h.sum() + mel_p.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))

def pick_position_from_brightness(bright):
    """
    Bridge -> bright (low pos); Neck -> dark (high pos).
    Map [0,1] brightness -> [0.1, 0.9] (inverted).
    """
    return float(np.clip(0.92 - 0.82 * bright, 0.1, 0.9))

def targets_from_npz(npz_dict, sr, hop, fmin_hz=30.0, fmax_hz=12000.0,
                     decay_min=0.08, decay_max=0.90,
                     attack_ms=40.0, attack_weight=0.4):
    """
    Build the 6D target vector from one .npz dictionary.

    Improvements for short sounds:
      - Compute 'brightness_attack' on a short window after onset_frame (≈40 ms).
      - If available, also compute attack brightness on the percussive mel (mel_p).
      - Fuse global and attack brightness to better reflect near-bridge picking.

    NOTE: decay_t60 is clamped into [decay_min, decay_max].
    """
    # Required features
    f0   = npz_dict["f0"].astype(float)
    vuv  = npz_dict["vuv"].astype(float)
    loud = npz_dict["loud"].astype(float)   # dB
    mel  = npz_dict["mel"].astype(float)    # (n_mels, T)

    # Optional features
    mel_h = npz_dict.get("mel_h", None)
    mel_p = npz_dict.get("mel_p", None)
    onset_frame = int(npz_dict.get("onset_frame", 0))

    # --- Core acoustic targets ---
    pitch_hz  = robust_median_f0(f0, vuv, fmin=20.0)

    decay_raw = estimate_t60_from_loudness_db(loud, sr, hop)
    decay_t60 = float(np.clip(decay_raw, decay_min, decay_max))

    # Global brightness over full note
    bright_global = spectral_brightness_ratio(mel)

    # Attack brightness over a short post-onset window
    win_frames = max(3, int(round((attack_ms / 1000.0) * sr / hop)))
    a_start = int(np.clip(onset_frame, 0, mel.shape[1] - 1))
    a_end   = int(np.clip(a_start + win_frames, a_start + 1, mel.shape[1]))

    bright_attack = spectral_brightness_ratio_slice(mel, a_start, a_end)

    # If percussive mel is present, take the max with its attack brightness
    if mel_p is not None and mel_p.size > 0:
        try:
            ba_p = spectral_brightness_ratio_slice(mel_p.astype(float), a_start, a_end)
            bright_attack = max(bright_attack, ba_p)
        except Exception:
            pass

    # Fuse global and attack brightness (heavier weight on global by default)
    w_att = float(np.clip(attack_weight, 0.0, 1.0))
    bright_mix = float(np.clip((1.0 - w_att) * bright_global + w_att * bright_attack, 0.0, 1.0))

    # Noise proportion from HPSS
    noise_mix = noise_mix_from_hn(mel_h, mel_p)

    # Heuristics
    # Higher decay -> lower damping; keep the same mapping.
    damping = float(np.clip(1.0 - (decay_t60 - 0.05) / (2.0 - 0.05), 0.0, 1.0))
    # Use mixed brightness (global+attack) to infer pick position
    pick_pos = pick_position_from_brightness(bright_mix)

    # Return vector in real units where applicable
    return np.array(
        [pitch_hz, decay_t60, bright_mix, damping, pick_pos, noise_mix],
        dtype=np.float32
    )
