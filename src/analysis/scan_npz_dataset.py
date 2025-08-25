# src/analysis/scan_npz_dataset.py
import argparse, yaml, csv, glob
from pathlib import Path
import numpy as np

EPS = 1e-8

def ema_smooth(x, alpha=0.2):
    if len(x) == 0:
        return x
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def estimate_noise_floor(loud_db, pre_frames=20):
    k = min(pre_frames, len(loud_db))
    return float(np.median(loud_db[:k])) if k > 0 else -80.0

def detect_onset_robust(loud_db, peak_drop_db=12.0, rise_over_floor_db=10.0):
    """
    Onset = first frame that is both:
      - within (peak - peak_drop_db) dB from the global peak
      - at least (noise_floor + rise_over_floor_db)
    """
    if len(loud_db) == 0:
        return 0
    ld = ema_smooth(loud_db, alpha=0.2)
    i_peak = int(np.argmax(ld)); Lp = float(ld[i_peak])
    floor = estimate_noise_floor(ld, pre_frames=20)
    thr = max(Lp - peak_drop_db, floor + rise_over_floor_db)
    cand = np.where(ld >= thr)[0]
    return int(cand[0]) if len(cand) else max(0, i_peak - 1)

def robust_pitch_in_window(f0, vuv, sr, hop, onset_idx, win_ms=(20, 250), fmin=20.0):
    T = len(f0)
    frames_per_ms = sr / hop / 1000.0
    i0 = int(onset_idx + win_ms[0] * frames_per_ms)
    i1 = int(onset_idx + win_ms[1] * frames_per_ms)
    i0 = max(0, min(T - 1, i0))
    i1 = max(i0 + 1, min(T, i1))
    mask = (vuv[i0:i1] > 0.5) & (f0[i0:i1] > fmin)
    if np.any(mask):
        return float(np.median(f0[i0:i1][mask]))
    # fallback to global voiced median
    mask_all = (vuv > 0.5) & (f0 > fmin)
    if np.any(mask_all):
        return float(np.median(f0[mask_all]))
    return 110.0

def spectral_brightness_ratio(mel):
    mel = np.atleast_2d(mel)
    n_mels = mel.shape[0]
    cutoff = int(n_mels * (2.0 / 3.0))
    top = mel[cutoff:, :]
    num = float(top.sum())
    den = float(mel.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))

def spectral_brightness_ratio_window(mel, i0, i1):
    mel = np.atleast_2d(mel)
    i0 = max(0, min(mel.shape[1]-1, i0))
    i1 = max(i0+1, min(mel.shape[1],   i1))
    seg = mel[:, i0:i1]
    n_mels = seg.shape[0]
    cutoff = int(n_mels * (2.0 / 3.0))
    top = seg[cutoff:, :]
    num = float(top.sum())
    den = float(seg.sum() + EPS)
    return float(np.clip(num / den, 0.0, 1.0))

def estimate_rt60_from_T20(loud_db, sr, hop, noise_guard_db=6.0):
    """
    RT60 via T20 (−5→−25 dB) with a guard above the noise floor:
    uses only frames louder than (floor + noise_guard_db).
    """
    if len(loud_db) < 8:
        return 0.3
    ld = ema_smooth(loud_db, alpha=0.2)
    i_peak = int(np.argmax(ld)); L0 = float(ld[i_peak])
    floor = estimate_noise_floor(ld, pre_frames=20)

    t = np.arange(len(ld)) * (hop / sr)
    x = t[i_peak:] - t[i_peak]
    y = (L0 - ld[i_peak:])  # dB drop from peak

    valid = ld[i_peak:] >= (floor + noise_guard_db)
    m = (y >= 5.0) & (y <= 25.0) & valid
    if np.sum(m) < 3:
        m = (y >= 0.0) & (y <= 25.0) & valid
        if np.sum(m) < 3:
            return 0.3

    A = np.vstack([x[m], np.ones(np.sum(m))]).T
    slope, _ = np.linalg.lstsq(A, y[m], rcond=None)[0]  # dB/s
    if slope <= 1e-6:
        return 0.3

    T20 = 20.0 / slope
    RT60 = 3.0 * T20
    return float(np.clip(RT60, 0.05, 2.5))

def load_cfg(config_path):
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def resolve_npz_dir(args, cfg):
    # prefer CLI npz_dir; else try cfg.paths.npz_out; else cfg.paths.npz_dir
    if args.npz_dir:
        return Path(args.npz_dir)
    p = None
    try:
        p = cfg["paths"]["npz_out"]
    except Exception:
        try:
            p = cfg["paths"]["npz_dir"]
        except Exception:
            p = None
    if p is None:
        raise ValueError("npz_dir not provided and not found in config (paths.npz_out / paths.npz_dir).")
    return Path(p)

def resolve_sr_hop(args, cfg):
    sr = args.sr if args.sr is not None else cfg.get("sample_rate", None)
    if sr is None:
        # sometimes nested under audio
        sr = cfg.get("audio", {}).get("sample_rate", None)
    hop = args.hop if args.hop is not None else cfg.get("hop_size", None)
    if hop is None:
        hop = cfg.get("audio", {}).get("hop_size", None)
    if sr is None or hop is None:
        # sensible defaults
        sr = sr or 44100
        hop = hop or 256
    return int(sr), int(hop)

def load_prompts(meta_csv):
    prom = {}
    if not meta_csv:
        return prom
    path = Path(meta_csv)
    if not path.exists():
        return prom
    # expect headers: name,prompt (or id,prompt). We'll accept any header with 'prompt'.
    import pandas as pd
    df = pd.read_csv(path)
    name_col = None
    for c in df.columns:
        if c.lower() in ("name", "id", "filename", "file", "stem"):
            name_col = c; break
    if name_col is None:
        # try infer name from filename if present
        if "wav" in df.columns:
            df["name"] = df["wav"].apply(lambda s: Path(s).stem)
            name_col = "name"
        else:
            # fallback: assume first column is name
            name_col = df.columns[0]
    prompt_col = None
    for c in df.columns:
        if "prompt" in c.lower():
            prompt_col = c; break
    if prompt_col is None:
        # create an empty
        df["prompt"] = ""
        prompt_col = "prompt"
    for _, row in df.iterrows():
        prom[str(row[name_col]).strip()] = str(row[prompt_col])
    return prom

def scan_one(npz_path, sr, hop, late_peak_thr, min_voiced_ratio, bright_low_thr, rt60_lo, rt60_hi):
    d = np.load(npz_path)
    mel   = d["mel"].astype(float)
    f0    = d["f0"].astype(float)
    vuv   = d["vuv"].astype(float)
    loud  = d["loud"].astype(float)
    mel_h = d.get("mel_h", None)
    mel_p = d.get("mel_p", None)

    T = len(loud)
    duration_s = T * (hop / sr)
    i_peak = int(np.argmax(loud))
    peak_pos = i_peak / max(1, T)

    onset = detect_onset_robust(loud, peak_drop_db=12.0, rise_over_floor_db=10.0)
    onset_pos = onset / max(1, T)

    voiced_ratio = float(np.mean(vuv > 0.5))

    pitch_win = robust_pitch_in_window(f0, vuv, sr, hop, onset_idx=onset, win_ms=(20, 250), fmin=20.0)
    bright_global = spectral_brightness_ratio(mel)
    # bright in attack window 30–120 ms after onset
    frames_per_ms = sr / hop / 1000.0
    i0_b = int(onset + 30 * frames_per_ms)
    i1_b = int(onset + 120 * frames_per_ms)
    bright_attack = spectral_brightness_ratio_window(mel, i0_b, i1_b)

    rt60_t20 = estimate_rt60_from_T20(loud, sr, hop, noise_guard_db=6.0)

    # warnings
    warns = []
    if peak_pos > late_peak_thr:
        warns.append(f"late_peak({peak_pos:.2f})")
    if voiced_ratio < min_voiced_ratio:
        warns.append(f"low_voiced({voiced_ratio:.2f})")
    if bright_attack < bright_low_thr:
        warns.append(f"low_bright_attack({bright_attack:.2f})")
    if rt60_t20 < rt60_lo:
        warns.append(f"rt60_short({rt60_t20:.2f}s)")
    if rt60_t20 > rt60_hi:
        warns.append(f"rt60_long({rt60_t20:.2f}s)")

    return {
        "name": Path(npz_path).stem,
        "T_frames": T,
        "sr": sr,
        "hop": hop,
        "duration_s": round(duration_s, 5),
        "loud_max_db": float(np.max(loud)),
        "peak_frame": i_peak,
        "peak_pos": round(peak_pos, 4),
        "onset_frame": onset,
        "onset_pos": round(onset_pos, 4),
        "voiced_ratio": round(voiced_ratio, 4),
        "f0_med_voiced_win_hz": round(pitch_win, 3),
        "f0_max_hz": round(float(np.max(f0)), 3),
        "brightness_global": round(bright_global, 4),
        "brightness_attack": round(bright_attack, 4),
        "rt60_T20_s": round(rt60_t20, 4),
        "warnings": ";".join(warns),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to base_real.yaml to read npz_dir/sr/hop")
    ap.add_argument("--npz_dir", default=None, help="override npz dir (if not using config)")
    ap.add_argument("--sr", type=int, default=None)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--meta_csv", default=None, help="optional prompts csv to join")
    ap.add_argument("--out_csv", default="data/real/results/npz_scan.csv")
    ap.add_argument("--out_joined", default="data/real/results/npz_scan_with_prompts.csv")
    # thresholds
    ap.add_argument("--late_peak_thr", type=float, default=0.30)
    ap.add_argument("--min_voiced_ratio", type=float, default=0.10)
    ap.add_argument("--bright_low_thr", type=float, default=0.08)
    ap.add_argument("--rt60_lo", type=float, default=0.10)
    ap.add_argument("--rt60_hi", type=float, default=1.20)
    args = ap.parse_args()

    cfg = load_cfg(args.config) if args.config else {}
    npz_dir = resolve_npz_dir(args, cfg)
    sr, hop = resolve_sr_hop(args, cfg)

    files = sorted(glob.glob(str(npz_dir / "*.npz")))
    assert files, f"No .npz files found in {npz_dir}"

    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        rows.append(scan_one(f, sr, hop, args.late_peak_thr, args.min_voiced_ratio,
                             args.bright_low_thr, args.rt60_lo, args.rt60_hi))

    # write scan csv
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[WROTE] {out_path} ({len(rows)} rows)")

    # summary counts
    total = len(rows)
    n_late = sum(1 for r in rows if "late_peak" in r["warnings"])
    n_lowv = sum(1 for r in rows if "low_voiced" in r["warnings"])
    n_lowb = sum(1 for r in rows if "low_bright_attack" in r["warnings"])
    n_short = sum(1 for r in rows if "rt60_short" in r["warnings"])
    n_long = sum(1 for r in rows if "rt60_long" in r["warnings"])
    print(f"[SUMMARY] total={total} late_peak={n_late} low_voiced={n_lowv} "
          f"low_bright_attack={n_lowb} rt60_short={n_short} rt60_long={n_long}")

    # optional join with prompts
    if args.meta_csv:
        prompts = load_prompts(args.meta_csv)
        joined_path = Path(args.out_joined)
        with open(out_path, "r", encoding="utf-8") as fi, open(joined_path, "w", newline="", encoding="utf-8") as fo:
            rdr = csv.DictReader(fi)
            fns = ["name", "prompt"] + [c for c in rdr.fieldnames if c != "name"]
            w = csv.DictWriter(fo, fieldnames=fns)
            w.writeheader()
            for row in rdr:
                stem = row["name"]
                base = stem.replace(".npz", "")
                # names in prompts are usually without extension; try both
                pr = prompts.get(stem, prompts.get(base, ""))
                out = {"name": stem, "prompt": pr}
                out.update(row)
                w.writerow(out)
        print(f"[WROTE] {joined_path}")

if __name__ == "__main__":
    main()
