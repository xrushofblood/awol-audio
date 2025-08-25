# tools/trim_onset.py
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

EPS = 1e-8

def frame_loudness_db(x, frame, hop):
    # simple RMS-based loudness in dB
    n = max(frame, 1)
    win = np.hanning(frame)
    T = 1 + (len(x)-frame)//hop if len(x) >= frame else 1
    out = np.zeros(T, dtype=float)
    for i in range(T):
        s = i*hop
        e = s+frame
        if e > len(x):
            seg = x[s:]
            w   = np.hanning(len(seg))
            rms = np.sqrt((seg*w**2).mean() + EPS)
        else:
            seg = x[s:e]
            rms = np.sqrt((seg*win).mean()**2 + EPS)
        out[i] = 20*np.log10(rms + EPS)
    return out

def estimate_noise_floor(loud_db, pre_frames=20):
    k = min(pre_frames, len(loud_db))
    return float(np.median(loud_db[:k])) if k>0 else -80.0

def detect_onset_robust(loud_db, peak_drop_db=12.0, rise_over_floor_db=10.0):
    if len(loud_db) == 0:
        return 0
    i_peak = int(np.argmax(loud_db)); Lp = float(loud_db[i_peak])
    floor = estimate_noise_floor(loud_db, pre_frames=20)
    thr = max(Lp - peak_drop_db, floor + rise_over_floor_db)
    cand = np.where(loud_db >= thr)[0]
    return int(cand[0]) if len(cand) else max(0, i_peak-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",  default="data/real/raw")
    ap.add_argument("--out_dir", default="data/real/raw_trimmed")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--frame", type=int, default=1024)
    ap.add_argument("--hop",   type=int, default=256)
    ap.add_argument("--prepad_ms", type=float, default=15.0)
    ap.add_argument("--target_len_s", type=float, default=1.0)
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    print(f"[INFO] trimming {len(wavs)} files from {in_dir} -> {out_dir}")

    for w in wavs:
        x, sr = sf.read(w)
        if x.ndim > 1: x = x.mean(axis=1)
        if sr != args.sr:
            print(f"[WARN] {w.name}: sr={sr} != {args.sr}, skipping")
            continue

        loud = frame_loudness_db(x, args.frame, args.hop)
        onset_f = detect_onset_robust(loud, peak_drop_db=12.0, rise_over_floor_db=10.0)

        # convert frame onset to samples
        onset_samp = max(0, int(onset_f * args.hop) - int(args.prepad_ms * sr / 1000.0))
        end_samp   = min(len(x), onset_samp + int(args.target_len_s * sr))
        y = x[onset_samp:end_samp]
        if len(y) < int(0.2*sr):  # ensure min length
            pad = int(0.2*sr) - len(y)
            y = np.pad(y, (0, pad))

        out = out_dir / w.name
        sf.write(out, y, sr)
        print(f"[OK] {w.name}: onset_frame={onset_f}, wrote {out}")

if __name__ == "__main__":
    main()
