# src/analysis/features.py
# Purpose:
#   Extract frame-aligned audio features for AWOL-audio (Day 1):
#   - mel spectrogram (and HPSS-derived mel envelopes)
#   - loudness proxy (RMS dB)
#   - f0 + periodicity (torchcrepe)
#   - robust V/UV mask built from periodicity + energy gates, with optional head/tail gating
#   - optional interpolation of f0 across voiced frames and simple octave-lift
#
# Notes:
#   - All time series are aligned to mel's time dimension (T).
#   - The V/UV logic is config-driven (see `vuv:` section in configs/base.yaml).
#   - A tighter f0 search range (e.g., 80–500 Hz) avoids subharmonics on synthetic plucks.

import os
import yaml
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import soundfile as sf


# -----------------------------
# Feature helpers
# -----------------------------
def compute_mel(y, sr, n_mels, n_fft, hop, fmin, fmax):
    """Return mel spectrogram in linear power (float32)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        fmin=fmin, fmax=fmax, power=2.0
    )
    return S.astype(np.float32)


def compute_hpss(y, sr, n_fft, hop, kernel_size=31, power=2.0):
    """
    HPSS on |STFT|^power, then inverse-STFT to get rough
    'harmonic' and 'percussive' waveforms.
    Returns (y_harm, y_perc) as float32.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** power
    H, P = librosa.decompose.hpss(S, kernel_size=kernel_size)
    y_h = librosa.istft(H ** (1.0 / power), hop_length=hop, length=len(y))
    y_p = librosa.istft(P ** (1.0 / power), hop_length=hop, length=len(y))
    return y_h.astype(np.float32), y_p.astype(np.float32)


def compute_rms_db(y, hop):
    """
    Frame-wise loudness proxy using RMS dB aligned to hop length.
    (We use a 4*hop frame to stabilize the estimate.)
    """
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-8))
    return rms_db.astype(np.float32)


def compute_f0_torchcrepe(y, sr, hop, fmin, fmax):
    """
    Predict raw f0 and periodicity using torchcrepe.
    Returns (f0_raw, periodicity) as float32 arrays (T,).
    """
    import torch
    import torchcrepe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_t = torch.tensor(y, dtype=torch.float32, device=device)[None]

    # torchcrepe returns (B, T) tensors
    f0, periodicity = torchcrepe.predict(
        y_t, sr, hop, fmin, fmax,
        model="full", batch_size=1024, device=device, return_periodicity=True
    )

    f0 = f0[0].detach().cpu().numpy().astype(np.float32)
    periodicity = periodicity[0].detach().cpu().numpy().astype(np.float32)
    return f0, periodicity


# -----------------------------
# Utility
# -----------------------------
def pad_or_crop(x, target_len):
    """Pad or crop 1D/2D arrays to match target_len on the time dimension."""
    if x.ndim == 1:
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)))
        return x[:target_len]
    if x.ndim == 2:
        if x.shape[1] < target_len:
            x = np.pad(x, ((0, 0), (0, target_len - x.shape[1])))
        return x[:, :target_len]
    return x


# -----------------------------
# V/UV post-processing
# -----------------------------
def postprocess_voicing(f0_raw, periodicity, rms_db, cfg):
    """
    Build a robust V/UV mask for plucked tones by combining:
      - a periodicity threshold (TorchCrepe)
      - an energy gate in dB (RMS)
      - an optional head/tail gate for very quiet onsets/decays
      - removal of tiny voiced blips shorter than min_len
    Optionally interpolate f0 across voiced frames and apply a simple octave-lift.

    Config keys under cfg["vuv"] (with sensible defaults):
      periodicity_thr: float (e.g., 0.35)
      energy_db_thr:  float (e.g., -42.0)
      min_voiced_len: int   (e.g., 3)
      head_tail_db:   float (e.g., -35.0)
      head_tail_win:  int   (e.g., 4)
      interpolate:    bool  (default True)
      octave_lift:    bool  (default True)
    """
    vuv_cfg = cfg.get("vuv", {})
    thr_p = float(vuv_cfg.get("periodicity_thr", 0.35))
    thr_e = float(vuv_cfg.get("energy_db_thr", -42.0))
    min_len = int(vuv_cfg.get("min_voiced_len", 3))
    head_tail_db = float(vuv_cfg.get("head_tail_db", -35.0))
    head_tail_win = int(vuv_cfg.get("head_tail_win", 4))
    do_interp = bool(vuv_cfg.get("interpolate", True))
    do_octave = bool(vuv_cfg.get("octave_lift", True))

    T = len(rms_db)

    # 1) Base voiced mask from periodicity and energy gate
    vuv = ((periodicity >= thr_p) & (rms_db >= thr_e)).astype(np.float32)

    # 2) Head/tail gating: if very quiet at the extremes, mark as unvoiced
    if head_tail_win > 0:
        # Head
        h_end = min(head_tail_win, T)
        if h_end > 0 and np.all(rms_db[:h_end] < head_tail_db):
            vuv[:h_end] = 0.0
        # Tail
        t_start = max(0, T - head_tail_win)
        if t_start < T and np.all(rms_db[t_start:] < head_tail_db):
            vuv[t_start:] = 0.0

    # 3) Remove tiny voiced blips shorter than min_len (run-length smoothing)
    if min_len > 1 and T > 0:
        start = 0
        while start < T:
            val = vuv[start]
            end = start + 1
            while end < T and vuv[end] == val:
                end += 1
            if val > 0.5 and (end - start) < min_len:
                vuv[start:end] = 0.0
            start = end

    # 4) Interpolate f0 across voiced frames only (optional)
    f0_out = f0_raw.astype(np.float32).copy()
    if do_interp:
        idx = np.where((f0_raw > 0) & (vuv > 0))[0]
        if len(idx) > 1:
            f0_interp = np.interp(np.arange(T), idx, f0_raw[idx]).astype(np.float32)
            f0_out = np.where(vuv > 0, f0_interp, 0.0).astype(np.float32)
        else:
            f0_out = np.where(vuv > 0, f0_raw, 0.0).astype(np.float32)
    else:
        f0_out = np.where(vuv > 0, f0_out, 0.0).astype(np.float32)

    # 5) Simple octave-lift to avoid subharmonics (optional)
    if do_octave:
        fmin_hz = float(cfg["f0"]["fmin_hz"])
        fmax_hz = float(cfg["f0"]["fmax_hz"])
        voiced_idx = np.where(vuv > 0)[0]
        for i in voiced_idx:
            f = f0_out[i]
            # Lift by octaves while too low (1.9 guards against boundary ping-pong)
            while 0 < f < (fmin_hz / 1.9):
                f *= 2.0
            # Clamp extreme highs (rare with a tight range)
            if f > fmax_hz * 1.2:
                f = fmax_hz
            f0_out[i] = f

    return f0_out.astype(np.float32), vuv.astype(np.float32)


# -----------------------------
# Main extraction routine
# -----------------------------
def extract_all(cfg):
    # Global analysis params
    sr = cfg["sample_rate"]
    n_fft = cfg["frame_size"]
    hop = cfg["hop_size"]
    n_mels = cfg["n_mels"]
    fmin = cfg["fmin_hz"]
    fmax = cfg["fmax_hz"]

    # f0 params
    f0_fmin_hz = float(cfg["f0"]["fmin_hz"])
    f0_fmax_hz = float(cfg["f0"]["fmax_hz"])

    # Paths
    proc_dir = cfg["paths"]["proc_dir"]
    meta_csv = cfg["paths"]["meta_csv"]
    out_dir = cfg["paths"]["npz_out"]
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(meta_csv)
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Features"):
        wav = os.path.join(proc_dir, row["filename"])
        if not os.path.exists(wav):
            print(f"[WARN] Missing processed wav: {wav}")
            continue

        y, _sr = sf.read(wav, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1).astype(np.float32)
        assert _sr == sr, f"Unexpected sample rate: {_sr} != {sr}"

        # 1) Mel (power) + HPSS waveforms
        mel = compute_mel(y, sr, n_mels, n_fft, hop, fmin, fmax)
        y_h, y_p = compute_hpss(
            y, sr, n_fft, hop,
            kernel_size=cfg["hpss"]["kernel_size"],
            power=cfg["hpss"]["power"]
        )

        # 2) Loudness proxy (RMS dB) aligned to hop
        rms_db = compute_rms_db(y, hop)

        # 3) Raw f0 + periodicity (no masking here)
        f0_raw, periodicity = compute_f0_torchcrepe(
            y, sr, hop, f0_fmin_hz, f0_fmax_hz
        )

        # 4) Mel envelopes for HPSS components
        mel_h = compute_mel(y_h, sr, n_mels, n_fft, hop, fmin, fmax)
        mel_p = compute_mel(y_p, sr, n_mels, n_fft, hop, fmin, fmax)

        # 5) Time alignment: pad/crop everything to T = mel.shape[1]
        T = mel.shape[1]
        rms_db = pad_or_crop(rms_db, T)
        f0_raw = pad_or_crop(f0_raw, T)
        periodicity = pad_or_crop(periodicity, T)
        mel_h = pad_or_crop(mel_h, T)
        mel_p = pad_or_crop(mel_p, T)

        # 6) V/UV post-processing + f0 interpolation/octave lift
        f0, vuv = postprocess_voicing(f0_raw, periodicity, rms_db, cfg)

        # 7) Save NPZ (one per clip)
        out_path = os.path.join(out_dir, row["filename"].replace(".wav", ".npz"))
        np.savez_compressed(
            out_path,
            prompt=str(row["prompt"]),
            f0=f0, vuv=vuv, loud=rms_db,     # 'loud' stores our RMS dB proxy
            mel=mel, mel_h=mel_h, mel_p=mel_p,
            sr=sr, hop=hop, n_mels=n_mels,
            fmin=fmin, fmax=fmax
        )
        # print(f"[OK] {row['filename']} -> {os.path.basename(out_path)}")


# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    extract_all(cfg)
