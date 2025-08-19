# src/analysis/inspect_npz.py
import os
import glob
import argparse
import numpy as np
import yaml
#import matplotlib.pyplot as plt

def describe_array(name, x, is_2d_ok=False):
    if x is None:
        return f"{name}: MISSING"
    if is_2d_ok:
        mn, mx = float(np.min(x)), float(np.max(x))
        return f"{name}:     shape={x.shape}, min={mn:.3f}, max={mx:.3f}"
    else:
        mn, mx = float(np.min(x)), float(np.max(x))
        return f"{name}:      shape={x.shape}, min={mn:.3f}, max={mx:.3f}"

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def micro_checks(npz, fname, cfg, loud_decay_window=10):
    """
    Lightweight consistency checks tailored for short, decay-like plucks.
    Produces a list of human-readable warnings (empty if all good).
    """
    warn = []

    sr   = npz.get("sr", None)
    hop  = npz.get("hop", None)
    mel  = npz.get("mel", None)
    f0   = npz.get("f0", None)
    vuv  = npz.get("vuv", None)
    loud = npz.get("loud", None)

    # ---- Basic presence / dtype
    for key in ["mel", "f0", "vuv", "loud"]:
        if key not in npz:
            warn.append(f"[{fname}] Missing key: {key}")
    if mel is None or f0 is None or vuv is None or loud is None:
        return warn

    # ---- Time alignment (T should match across 1D tracks and mel's second dim)
    T = mel.shape[1]
    for name, arr in [("f0", f0), ("vuv", vuv), ("loud", loud)]:
        if arr.ndim != 1:
            warn.append(f"[{fname}] {name} is not 1D (ndim={arr.ndim}).")
        if len(arr) != T:
            warn.append(f"[{fname}] Time mismatch: {name}.len={len(arr)} vs mel.T={T}")

    # ---- Value ranges and validity
    if np.any(np.isnan(mel)) or np.any(np.isinf(mel)):
        warn.append(f"[{fname}] mel contains NaN/Inf.")
    if np.any(np.isnan(f0)) or np.any(np.isinf(f0)):
        warn.append(f"[{fname}] f0 contains NaN/Inf.")
    if np.any(np.isnan(vuv)) or np.any(np.isinf(vuv)):
        warn.append(f"[{fname}] vuv contains NaN/Inf.")
    if np.any(np.isnan(loud)) or np.any(np.isinf(loud)):
        warn.append(f"[{fname}] loud contains NaN/Inf.")

    # ---- vuv sanity: should be {0,1} (float ok) and not trivial in aggregate
    v_unique = np.unique(vuv)
    if not np.all(np.isin(v_unique, [0.0, 1.0])):
        warn.append(f"[{fname}] vuv has values outside {{0,1}}: unique={v_unique[:6]}")
    voiced_ratio = float(np.mean(vuv > 0.5))
    if voiced_ratio < 0.05:
        warn.append(f"[{fname}] vuv ~all unvoiced (voiced_ratio={voiced_ratio:.3f}).")
    if voiced_ratio > 0.95:
        # For a synthetic pluck this can be fine, just emit info-level hint
        warn.append(f"[{fname}] vuv ~all voiced (voiced_ratio={voiced_ratio:.3f}) – OK for clean plucks.")

    # ---- f0 vs vuv consistency
    # Expect f0 ~ 0 when unvoiced; > 0 when voiced (allow tiny epsilon)
    eps = 1e-6
    uv_idx = np.where(vuv < 0.5)[0]
    v_idx  = np.where(vuv >= 0.5)[0]
    if uv_idx.size > 0:
        frac_uv_nonzero_f0 = float(np.mean(f0[uv_idx] > eps))
        if frac_uv_nonzero_f0 > 0.05:
            warn.append(f"[{fname}] {frac_uv_nonzero_f0*100:.1f}% of unvoiced frames have nonzero f0.")
    if v_idx.size > 0:
        frac_v_zero_f0 = float(np.mean(f0[v_idx] <= eps))
        if frac_v_zero_f0 > 0.05:
            warn.append(f"[{fname}] {frac_v_zero_f0*100:.1f}% of voiced frames have zero f0.")

    # ---- Loudness trend for pluck: peak near start, then decay (heuristic)
    # We'll check that the global max is in the first third, and that a median
    # over the last window is below the max by some margin.
    if T >= max(8, loud_decay_window):
        peak_idx = int(np.argmax(loud))
        if peak_idx > T // 3:
            warn.append(f"[{fname}] loudness peak at frame {peak_idx} (>{T//3}); atypical for a pluck.")
        tail = loud[-loud_decay_window:]
        if np.median(tail) > (np.max(loud) - 3.0):
            # still quite loud at the end, maybe trailing silence missing?
            warn.append(f"[{fname}] loudness tail median not clearly below peak (possible missing fade/silence).")

    # ---- Mel energy non-negativity (power mel expected >= 0)
    if np.min(mel) < 0.0:
        warn.append(f"[{fname}] mel has negative values; check mel scale (power vs dB).")

    # ---- Hop/sample-rate consistency
    if sr is not None and hop is not None:
        # duration (s) for rough sanity output in caller
        pass

    return warn

def pretty_print_header():
    print()

def pretty_print_file(npz, fname):
    mel  = npz["mel"]; f0 = npz["f0"]; vuv = npz["vuv"]; loud = npz["loud"]
    n_mels = int(npz.get("n_mels", mel.shape[0]))
    sr   = int(npz.get("sr", -1))
    hop  = int(npz.get("hop", -1))

    print(f"[FILE] {fname}")
    print(describe_array("  mel", mel, is_2d_ok=True))
    T = mel.shape[1]
    print(describe_array("  f0", f0))
    print(f"           (T={len(f0)}, mel.T={T})")
    print(describe_array("  vuv", vuv))
    print(describe_array("  loud", loud))
    print(describe_array("  mel_h", npz.get("mel_h"), is_2d_ok=True) if "mel_h" in npz else "  mel_h:     (absent)")
    print(describe_array("  mel_p", npz.get("mel_p"), is_2d_ok=True) if "mel_p" in npz else "  mel_p:     (absent)")
    print(f"  sr={sr}, hop={hop}, n_mels={n_mels}")
    print()

'''def maybe_plot(npz, title, max_frames=200):
    """Optional quick-look plot for a few aligned tracks."""
    mel  = npz["mel"]; f0 = npz["f0"]; vuv = npz["vuv"]; loud = npz["loud"]
    T = mel.shape[1]
    t = np.arange(T)

    # select a reasonable window
    end = min(T, max_frames)
    sl = slice(0, end)

    plt.figure()
    plt.title(f"{title} — f0 / vuv / loud (first {end} frames)")
    plt.plot(t[sl], f0[sl], label="f0 (Hz)")
    plt.plot(t[sl], vuv[sl] * np.nanmax(f0[sl] + 1e-6), label="vuv (scaled)")
    plt.plot(t[sl], loud[sl], label="loud (dB)")
    plt.xlabel("frame idx")
    plt.legend()
    plt.tight_layout()
    plt.show()'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    ap.add_argument("--limit", type=int, default=5, help="Max number of files to print.")
    ap.add_argument("--glob", default="*.npz", help="Filename glob to filter inspected files.")
    ap.add_argument("--plot", action="store_true", help="Plot f0/vuv/loud for each inspected file.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    npz_dir = cfg["paths"]["npz_out"]
    pattern = os.path.join(npz_dir, args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No .npz found in: {pattern}")
        return

    to_show = files[:args.limit] if args.limit > 0 else files
    all_warnings = []

    for fp in to_show:
        fname = os.path.basename(fp)
        with np.load(fp, allow_pickle=True) as data:
            # Print summary
            pretty_print_file(data, fname)
            # Micro-checks
            warnings = micro_checks(data, fname, cfg)
            all_warnings.extend(warnings)
            for w in warnings:
                print("  [WARN]", w)
           # if args.plot:
                #maybe_plot(data, fname)

    # Aggregate quick report
    print("\n=== Micro-check summary ===")
    if not all_warnings:
        print("No warnings emitted. Data looks consistent for pluck-style signals.")
    else:
        # Group by message (deduplicate but keep counts)
        from collections import Counter
        counts = Counter(all_warnings)
        for msg, c in counts.most_common():
            print(f"({c}x) {msg}")

if __name__ == "__main__":
    main()
