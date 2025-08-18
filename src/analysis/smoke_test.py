import os
import csv
import argparse
import soundfile as sf
import numpy as np
import yaml

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def check_unique_filenames(rows):
    names = [r["filename"] for r in rows]
    assert len(names) == len(set(names)), "Duplicate filenames found in prompts.csv"

def check_prompts_nonempty(rows):
    for r in rows:
        p = (r.get("prompt") or "").strip()
        assert len(p) > 0, f"Empty prompt for filename={r.get('filename')}"

def check_files_exist(rows, raw_dir):
    missing = [r["filename"] for r in rows if not os.path.exists(os.path.join(raw_dir, r["filename"]))]
    assert len(missing) == 0, f"Missing WAV files in data/raw: {missing[:5]} ... (total {len(missing)})"

def check_only_listed_exist(rows, raw_dir):
    listed = set([r["filename"] for r in rows])
    extra = [f for f in os.listdir(raw_dir) if f.lower().endswith(".wav") and f not in listed]
    assert len(extra) == 0, f"Found WAV files not listed in prompts.csv: {extra[:5]} ... (total {len(extra)})"

def check_audio_properties(rows, raw_dir, sr_expected, dur_target_s=1.0, dur_tol_s=0.15, target_peak_db=-1.0):
    # Duration window: e.g., 0.85s–1.15s (tolleranza per sicurezza)
    min_d = dur_target_s - dur_tol_s
    max_d = dur_target_s + dur_tol_s

    bad_sr, bad_len, silent, clipped = [], [], [], []
    for r in rows:
        fp = os.path.join(raw_dir, r["filename"])
        y, sr = sf.read(fp, always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != sr_expected:
            bad_sr.append((r["filename"], sr))
        dur = len(y) / float(sr)
        if not (min_d <= dur <= max_d):
            bad_len.append((r["filename"], round(dur, 3)))

        peak = np.max(np.abs(y)) if len(y) else 0.0
        if peak < 1e-4:
            silent.append(r["filename"])

        # check near-target peak (non rigido: controlla che non sia >1 o << -1dB)
        if peak > 1.0 + 1e-6:
            clipped.append(r["filename"])

    assert len(bad_sr) == 0, f"Unexpected sample rate: {bad_sr[:5]} ... (total {len(bad_sr)})"
    assert len(bad_len) == 0, f"Unexpected duration: {bad_len[:5]} ... (total {len(bad_len)})"
    assert len(silent) == 0, f"Silent or near-silent files: {silent[:5]} ... (total {len(silent)})"
    assert len(clipped) == 0, f"Clipped files (peak>1.0): {clipped[:5]} ... (total {len(clipped)})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["paths"]["raw_dir"]
    meta_csv = cfg["paths"]["meta_csv"]
    sr_expected = cfg["sample_rate"]
    target_peak_db = cfg.get("target_peak_db", -1.0)

    assert os.path.exists(raw_dir), f"Missing directory: {raw_dir}"
    assert os.path.exists(meta_csv), f"Missing metadata CSV: {meta_csv}"

    rows = read_csv(meta_csv)
    assert len(rows) > 0, "prompts.csv has no rows"

    print(f"[SMOKE] Loaded {len(rows)} rows from {meta_csv}")
    check_unique_filenames(rows)
    check_prompts_nonempty(rows)
    check_files_exist(rows, raw_dir)
    check_only_listed_exist(rows, raw_dir)
    check_audio_properties(rows, raw_dir, sr_expected, dur_target_s=1.0, dur_tol_s=0.15, target_peak_db=target_peak_db)
    print("[SMOKE] All checks passed ")

if __name__ == "__main__":
    main()
