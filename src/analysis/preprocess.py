# src/analysis/preprocess.py

import os
import argparse
import yaml
import librosa
import soundfile as sf
import numpy as np
import csv
from pathlib import Path


def preprocess_audio(y, sr, cfg):
    """
    Optional trim -> resample -> peak normalize.
    Trimming is controlled by cfg['apply_trim'] (default: False).
    """
    # --- 0) Optional silence trim (NEW flag) ---
    apply_trim = bool(cfg.get("apply_trim", False))          # NEW
    trim_db = float(cfg.get("trim_db", 40.0))                # unchanged default

    if apply_trim:
        y_trim, _ = librosa.effects.trim(y, top_db=trim_db)
        # Guard against over-trim: if empty, keep original
        if y_trim.size > 0:
            y = y_trim

    # --- 1) Resample if needed (unchanged) ---
    target_sr = int(cfg.get("sample_rate", sr))
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # --- 2) Peak normalize (unchanged, made slightly safer) ---
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.0:
        target_peak_db = float(cfg.get("target_peak_db", -1.0))  # -1 dBFS
        target_peak = 10 ** (target_peak_db / 20.0)
        y = (y / peak) * target_peak

    return y, sr


def run_preprocess(cfg):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    proc_dir = Path(cfg["paths"]["proc_dir"])
    meta_csv = Path(cfg["paths"]["meta_csv"])

    proc_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        in_path = raw_dir / r["filename"]
        out_path = proc_dir / r["filename"]

        # Load as mono, preserve native sr (resampled inside preprocess_audio)
        y, sr = librosa.load(in_path, sr=None, mono=True)

        y, sr = preprocess_audio(y, sr, cfg)
        sf.write(out_path, y, sr)

        dur = (len(y) / sr) if sr > 0 else 0.0
        print(f"[OK] {in_path.name} → {out_path.name} ({dur:.2f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_preprocess(cfg)


if __name__ == "__main__":
    main()
