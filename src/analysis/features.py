# src/analysis/features.py
# Purpose:
#   Extract frame-aligned audio features for AWOL-audio Day 1:
#   - mel spectrogram (and HPSS-derived mel envelopes)
#   - loudness (RMS dB proxy)
#   - f0 + periodicity (torchcrepe)
#   - hybrid VUV mask built from periodicity and loudness
#
# Notes:
#   - We align all time-series to mel's number of frames (T).
#   - f0 is interpolated only on voiced frames (according to the hybrid VUV).
#   - The hybrid VUV = (periodicity > thr) OR (loudness > thr_db)
#     helps with transient/decaying plucks that are often marked as unvoiced.

import os
import yaml
import numpy as np
import pandas as pd
import librosa
import pyloudnorm as pyln  # kept for future LUFS use; here we use RMS dB as frame-wise proxy
from tqdm import tqdm
import soundfile as sf


# -----------------------------
# Feature helpers
# -----------------------------
def compute_mel(y, sr, n_mels, n_fft, hop, fmin, fmax):
    """Return linear-power mel spectrogram (float32)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        fmin=fmin, fmax=fmax, power=2.0
    )
    return S.astype(np.float32)


def compute_hpss(y, sr, n_fft, hop, kernel_size=31, power=2.0):
    """
    HPSS on magnitude STFT^power, then inverse-STFT to get
    rough 'harmonic' and 'percussive' waveforms.
    Returns (y_harm, y_perc) as float32.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** power
    H, P = librosa.decompose.hpss(S, kernel_size=kernel_size)
    y_h = librosa.istft(H ** (1.0 / power), hop_length=hop, length=len(y))
    y_p = librosa.istft(P ** (1.0 / power), hop_length=hop, length=len(y))
    return y_h.astype(np.float32), y_p.astype(np.float32)


def compute_loudness(y, sr, hop):
    """
    Frame-wise loudness proxy using RMS dB aligned to hop length.
    (pyln is imported for future LUFS needs; not used here.)
    """
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-8))
    return rms_db.astype(np.float32)


def compute_f0_torchcrepe(y, sr, hop, fmin, fmax):
    """
    Predict raw f0 and periodicity using torchcrepe.
    We return (f0_raw, periodicity) and build the final VUV in extract_all().
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
    """Pad or crop 1D/2D arrays to match target_len on time dimension."""
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
# Main extraction routine
# -----------------------------
def extract_all(cfg):
    sr = cfg["sample_rate"]
    n_fft = cfg["frame_size"]
    hop = cfg["hop_size"]
    n_mels = cfg["n_mels"]
    fmin = cfg["fmin_hz"]
    fmax = cfg["fmax_hz"]

    proc_dir = cfg["paths"]["proc_dir"]
    meta_csv = cfg["paths"]["meta_csv"]
    out_dir = cfg["paths"]["npz_out"]
    os.makedirs(out_dir, exist_ok=True)

    # thresholds for hybrid VUV
    # lower periodicity threshold helps on plucked transients
    periodicity_thr = 0.05
    loud_thr_db = -55.0

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

        # 1) mel (power) + HPSS waveforms
        mel = compute_mel(y, sr, n_mels, n_fft, hop, fmin, fmax)
        y_h, y_p = compute_hpss(
            y, sr, n_fft, hop,
            kernel_size=cfg["hpss"]["kernel_size"],
            power=cfg["hpss"]["power"]
        )

        # 2) loudness (RMS dB) aligned to hop
        loud = compute_loudness(y, sr, hop)

        # 3) raw f0 + periodicity (no masking here)
        f0_raw, periodicity = compute_f0_torchcrepe(
            y, sr, hop,
            cfg["f0"]["fmin_hz"], cfg["f0"]["fmax_hz"]
        )

        # 4) build mel envelopes for HPSS components
        mel_h = compute_mel(y_h, sr, n_mels, n_fft, hop, fmin, fmax)
        mel_p = compute_mel(y_p, sr, n_mels, n_fft, hop, fmin, fmax)

        # 5) time alignment: pad/crop everything to T = mel.shape[1]
        T = mel.shape[1]
        loud = pad_or_crop(loud, T)
        f0_raw = pad_or_crop(f0_raw, T)
        periodicity = pad_or_crop(periodicity, T)
        mel_h = pad_or_crop(mel_h, T)
        mel_p = pad_or_crop(mel_p, T)

        # 6) build hybrid VUV = (periodicity > thr) OR (loud > thr_db)
        vuv_crepe = (periodicity > periodicity_thr).astype(np.float32)
        vuv_energy = (loud > loud_thr_db).astype(np.float32)
        vuv = np.maximum(vuv_crepe, vuv_energy).astype(np.float32)

        # 7) interpolate f0 over voiced frames only (according to hybrid VUV)
        idx = np.where((f0_raw > 0) & (vuv > 0))[0]
        if len(idx) > 1:
            f0_interp = np.interp(np.arange(len(f0_raw)), idx, f0_raw[idx]).astype(np.float32)
        else:
            f0_interp = f0_raw.astype(np.float32)
        f0 = np.where(vuv > 0, f0_interp, 0.0).astype(np.float32)

        # 8) save NPZ (one per clip)
        out_path = os.path.join(out_dir, row["filename"].replace(".wav", ".npz"))
        np.savez_compressed(
            out_path,
            prompt=str(row["prompt"]),
            f0=f0, vuv=vuv, loud=loud,
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
