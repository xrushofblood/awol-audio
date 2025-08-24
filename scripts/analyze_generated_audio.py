# scripts/analyze_generated_audio.py
# Analyze generated audio files and export a CSV with simple descriptors.
# It first tries to import `extract_all_features` from src.analysis.spectral_feats.
# If unavailable, it falls back to local implementations (librosa-based).

import argparse
import csv
import glob
import os
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional

import numpy as np

# --- Optional imports (soft dependencies) ---
try:
    import soundfile as sf
except Exception:
    sf = None

# librosa is optional but strongly recommended for the fallback path
try:
    import librosa
    import librosa.feature
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

# -------------------------------
# Import strategy for extractor
# -------------------------------

def _import_external_extractor() -> Optional[Callable[[np.ndarray, int], Dict[str, float]]]:
    """
    Try to import `extract_all_features(y, sr)` from:
      1) src.analysis.spectral_feats
      2) analysis.spectral_feats  (in case src is project root)
      3) spectral_feats           (same folder as this script)
    Returns the callable if found, else None.
    """
    # 1) src.analysis.spectral_feats
    try:
        from src.analysis.spectral_feats import extract_all_features  # type: ignore
        return extract_all_features
    except Exception:
        pass

    # 2) analysis.spectral_feats
    try:
        from analysis.spectral_feats import extract_all_features  # type: ignore
        return extract_all_features
    except Exception:
        pass

    # 3) local spectral_feats.py (next to this script)
    try:
        from spectral_feats import extract_all_features  # type: ignore
        return extract_all_features
    except Exception:
        pass

    return None

# -------------------------------
# Local fallback implementations
# -------------------------------

def _safe_norm_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))

def _estimate_decay_times(y: np.ndarray, sr: int,
                          frac_start: float = 0.02,
                          frac_end: float = 0.95) -> Tuple[float, float]:
    """
    Rough T20/T40 decay estimation from the envelope in dB.
    - Finds linear slope (dB / second) on a middle chunk of the envelope,
      then estimates times to drop by 20 and 40 dB.
    Returns (T20_sec, T40_sec). 0.0 if cannot estimate.
    """
    if y.size == 0:
        return 0.0, 0.0

    # Envelope via absolute value + small smoothing
    env = np.abs(y)
    if env.max() < 1e-6:
        return 0.0, 0.0

    # Simple moving average (5 ms)
    win = max(3, int(0.005 * sr))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env_s = np.convolve(env, kernel, mode="same")
    edb = _safe_norm_db(env_s + 1e-12)

    n = edb.shape[0]
    i0 = int(frac_start * n)
    i1 = int(frac_end * n)
    if i1 <= i0 + 5:
        return 0.0, 0.0

    x_t = np.arange(i0, i1, dtype=np.float32) / float(sr)
    y_db = edb[i0:i1].astype(np.float32)

    # Robust linear fit
    x_mean = x_t.mean()
    y_mean = y_db.mean()
    num = np.sum((x_t - x_mean) * (y_db - y_mean))
    den = np.sum((x_t - x_mean) ** 2) + 1e-12
    slope_db_per_sec = num / den  # dB / sec

    if slope_db_per_sec >= -1e-6:
        # Non-decaying or increasing
        return 0.0, 0.0

    T20 = 20.0 / (-slope_db_per_sec)
    T40 = 40.0 / (-slope_db_per_sec)
    return float(max(T20, 0.0)), float(max(T40, 0.0))

def _median_or_zero(x: np.ndarray) -> float:
    if x.size == 0 or not np.isfinite(x).any():
        return 0.0
    return float(np.nanmedian(x[np.isfinite(x)]))

def _fallback_extract_all_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Minimal feature set compatible with spectral_feats.extract_all_features:
      - rms
      - f0_hz_med (librosa.yin if available, else 0.0)
      - spectral_centroid_hz_med
      - rolloff85_hz_med
      - decay_T20_sec, decay_T40_sec
      - duration_sec
    """
    feats: Dict[str, float] = {}

    # duration
    feats["duration_sec"] = float(len(y) / float(sr) if sr > 0 else 0.0)

    # RMS
    feats["rms"] = float(np.sqrt(np.mean(y**2)) if y.size else 0.0)

    # Pitch (median)
    f0_med = 0.0
    if _HAS_LIBROSA and y.size:
        try:
            # Use YIN; set a wide range
            f0 = librosa.yin(y, fmin=50, fmax=min(2000, sr/2 - 1), sr=sr)
            f0_med = _median_or_zero(f0[np.isfinite(f0)])
        except Exception:
            f0_med = 0.0
    feats["f0_hz_med"] = float(max(f0_med, 0.0))

    # Spectral features
    if _HAS_LIBROSA and y.size:
        try:
            S = np.abs(librosa.stft(y=y, n_fft=2048, hop_length=256, win_length=1024))
            freq = librosa.fft_frequencies(sr=sr, n_fft=2048)

            # Centroid (Hz) per frame, then median
            sc = librosa.feature.spectral_centroid(S=S, freq=freq)
            feats["spectral_centroid_hz_med"] = _median_or_zero(sc)

            # Rolloff 85% (Hz) per frame, then median
            ro = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
            feats["rolloff85_hz_med"] = _median_or_zero(ro)
        except Exception:
            feats["spectral_centroid_hz_med"] = 0.0
            feats["rolloff85_hz_med"] = 0.0
    else:
        feats["spectral_centroid_hz_med"] = 0.0
        feats["rolloff85_hz_med"] = 0.0

    # Decay times
    T20, T40 = _estimate_decay_times(y, sr)
    feats["decay_T20_sec"] = float(T20)
    feats["decay_T40_sec"] = float(T40)

    return feats

# -------------------------------
# I/O helpers
# -------------------------------

def _load_wav(path: Path, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load mono waveform. Prefer soundfile; fallback to librosa.
    If target_sr is provided and librosa is available, it will resample.
    """
    if sf is not None:
        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if target_sr is not None and _HAS_LIBROSA and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return y, int(sr)
    else:
        if not _HAS_LIBROSA:
            raise RuntimeError("No soundfile nor librosa available to read audio.")
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        y = y.astype(np.float32)
        return y, int(sr)

def _guess_prompt_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    noext = os.path.splitext(base)[0]
    return noext.replace("_", " ").replace("-", " ").strip()

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze generated audio files and export CSV metrics.")
    ap.add_argument("--input_dir", required=True, help="Directory with generated WAVs")
    ap.add_argument("--glob", default="*.wav", help="Glob pattern for files (default: *.wav)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate for analysis (default: 16000)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([Path(p) for p in glob.glob(str(in_dir / args.glob))])
    if not files:
        print(f"[WARN] No files found in {in_dir} matching {args.glob}")
    else:
        print(f"[INFO] Found {len(files)} files to analyze.")

    # Choose extractor: external (spectral_feats) or fallback
    external_extractor = _import_external_extractor()
    if external_extractor is not None:
        print("[INFO] Using external extractor: src.analysis.spectral_feats.extract_all_features")
        extractor_fn = external_extractor
    else:
        print("[INFO] Using fallback local extractor (librosa-based).")
        extractor_fn = _fallback_extract_all_features

    header = [
        "file",
        "prompt_guess",
        "sr",
        "rms",
        "f0_hz_med",
        "spectral_centroid_hz_med",
        "rolloff85_hz_med",
        "decay_T20_sec",
        "decay_T40_sec",
        "duration_sec",
        "ok",
        "has_content",
    ]

    rows = []
    for fp in files:
        try:
            y, sr = _load_wav(fp, target_sr=args.sr)
            has_content = bool(np.any(np.abs(y) > 1e-6))
            if not has_content:
                feats = {
                    "rms": 0.0,
                    "f0_hz_med": 0.0,
                    "spectral_centroid_hz_med": 0.0,
                    "rolloff85_hz_med": 0.0,
                    "decay_T20_sec": 0.0,
                    "decay_T40_sec": 0.0,
                    "duration_sec": float(len(y) / float(sr) if sr > 0 else 0.0),
                }
            else:
                feats = extractor_fn(y, sr)

            # Clean any NaN/Inf to 0.0
            for k, v in list(feats.items()):
                if not np.isfinite(v):
                    feats[k] = 0.0

            row = {
                "file": fp.name,
                "prompt_guess": _guess_prompt_from_filename(fp.name),
                "sr": int(sr),
                "rms": float(feats.get("rms", 0.0)),
                "f0_hz_med": float(feats.get("f0_hz_med", 0.0)),
                "spectral_centroid_hz_med": float(feats.get("spectral_centroid_hz_med", 0.0)),
                "rolloff85_hz_med": float(feats.get("rolloff85_hz_med", 0.0)),
                "decay_T20_sec": float(feats.get("decay_T20_sec", 0.0)),
                "decay_T40_sec": float(feats.get("decay_T40_sec", 0.0)),
                "duration_sec": float(feats.get("duration_sec", len(y) / float(sr) if sr > 0 else 0.0)),
                "ok": True,
                "has_content": bool(has_content),
            }
        except Exception as e:
            row = {
                "file": fp.name,
                "prompt_guess": _guess_prompt_from_filename(fp.name),
                "sr": 0,
                "rms": 0.0,
                "f0_hz_med": 0.0,
                "spectral_centroid_hz_med": 0.0,
                "rolloff85_hz_med": 0.0,
                "decay_T20_sec": 0.0,
                "decay_T40_sec": 0.0,
                "duration_sec": 0.0,
                "ok": False,
                "has_content": False,
            }
            print(f"[ERR] {fp.name}: {e}")

        rows.append(row)

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(f, fieldnames=header)
        cw.writeheader()
        for r in rows:
            cw.writerow(r)

    print(f"[DONE] Wrote: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
