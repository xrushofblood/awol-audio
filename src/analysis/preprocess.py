import os
import argparse
import yaml
import librosa
import soundfile as sf
import numpy as np
import csv
from pathlib import Path

def preprocess_audio(y, sr, cfg):
    # Resample if needed
    target_sr = cfg["sample_rate"]
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=cfg.get("trim_db", 40))

    # Normalize peak
    peak = np.max(np.abs(y))
    if peak > 0:
        target_peak = 10 ** (cfg.get("target_peak_db", -1.0) / 20)
        y = y / peak * target_peak

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

        y, sr = librosa.load(in_path, sr=None, mono=True)
        y, sr = preprocess_audio(y, sr, cfg)
        sf.write(out_path, y, sr)

        print(f"[OK] {in_path.name} → {out_path.name} ({len(y)/sr:.2f}s)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_preprocess(cfg)

if __name__ == "__main__":
    main()