# src/synth/evaluate_params.py
import argparse, yaml, glob, math
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from .param_regressor import ParamRegressor
from .targets_from_npz import targets_from_npz

EPS = 1e-8

def l2norm_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + EPS
    return x / n

def load_perfile_embeddings(audio_emb_dir):
    files = sorted(glob.glob(str(Path(audio_emb_dir) / "*.audio.npy")))
    assert files, f"No *.audio.npy found in {audio_emb_dir}"
    X = [np.load(f).astype("float32") for f in files]
    names = [Path(f).stem.replace(".audio", "") for f in files]
    return np.stack(X, axis=0), names

def to_unit(x, spec):
    """
    Map real-valued target -> [0,1] using spec {min,max,scale?}.
    scale: 'linear' (default) or 'log_hz' (for pitch in Hz).
    """
    lo, hi = float(spec["min"]), float(spec["max"])
    scale = spec.get("scale", "linear")
    x = np.asarray(x, dtype=np.float32)
    if scale == "log_hz":
        x = np.log2(np.clip(x, lo, hi))
        lo_log, hi_log = math.log2(lo), math.log2(hi)
        return (x - lo_log) / (hi_log - lo_log + 1e-9)
    else:
        return (x - lo) / (hi - lo + 1e-9)

def from_unit(u, spec):
    """Map normalized [0,1] -> real-valued target using spec {min,max,scale?}."""
    lo, hi = float(spec["min"]), float(spec["max"])
    scale = spec.get("scale", "linear")
    u = np.asarray(u, dtype=np.float32)
    u = np.clip(u, 0.0, 1.0)
    if scale == "log_hz":
        lo_log, hi_log = math.log2(lo), math.log2(hi)
        x_log = lo_log + u * (hi_log - lo_log)
        return (2.0 ** x_log).astype(np.float32)
    else:
        return (lo + u * (hi - lo)).astype(np.float32)

def get_spec(cfg, name):
    for s in cfg["synth"]["params"]:
        if s["name"] == name:
            return s
    raise KeyError(f"Param spec '{name}' not found")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params_real.yaml")
    ap.add_argument("--ckpt",   default="checkpoints/paramreg/paramreg_best.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # specs (ordered) from config
    specs = cfg["synth"]["params"]
    names_order = [p["name"] for p in specs]

    # get decay spec to clamp labels
    decay_spec = get_spec(cfg, "decay_t60")
    dmin, dmax = float(decay_spec["min"]), float(decay_spec["max"])

    # --- build dataset (X in CLAP space, Y in real units) ---
    X, names = load_perfile_embeddings(cfg["paths"]["audio_emb_dir"])
    X = l2norm_np(X)

    Y_list = []
    for n in names:
        d = np.load(Path(cfg["paths"]["npz_dir"]) / f"{n}.npz")
        y = targets_from_npz(
            d,
            sr=cfg["audio"]["sample_rate"],
            hop=cfg["audio"]["hop_size"],
            fmin_hz=cfg["audio"].get("fmin_hz", 30.0),
            fmax_hz=cfg["audio"].get("fmax_hz", 12000.0),
            decay_min=dmin, decay_max=dmax,   # <-- clamp here
        )
        Y_list.append(y)
    Y_real = np.stack(Y_list, axis=0).astype("float32")  # (N, P)

    # normalize targets to [0,1] per spec
    Y01 = np.zeros_like(Y_real, dtype=np.float32)
    for j, spec in enumerate(specs):
        Y01[:, j] = np.clip(to_unit(Y_real[:, j], spec), 0.0, 1.0)

    # split
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=cfg["train"]["val_ratio"],
        random_state=cfg["train"]["seed"],
        shuffle=True,
    )
    Xv, Yv_real, Yv01 = X[va_idx], Y_real[va_idx], Y01[va_idx]

    # --- load model ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ParamRegressor(
        in_dim=cfg["model"]["in_dim"],
        hidden=cfg["model"]["hidden"],
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
    )
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred01 = model(torch.from_numpy(Xv)).cpu().numpy()  # (B,P) in [0,1]

    # denormalize predictions to real units per spec
    pred_real_cols = []
    for j, spec in enumerate(specs):
        pred_real_cols.append(from_unit(pred01[:, j], spec))
    pred_real = np.stack(pred_real_cols, axis=1)

    # indices for pitch/decay if present
    def safe_idx(name):
        return names_order.index(name) if name in names_order else None

    pitch_idx = safe_idx("pitch_hz")
    decay_idx = safe_idx("decay_t60")

    # --- metrics ---
    def mae(a, b): return float(np.mean(np.abs(a - b)))
    def corr(a, b):
        if a.ndim == 1: a = a.reshape(-1)
        if b.ndim == 1: b = b.reshape(-1)
        if len(a) < 2: return float("nan")
        C = np.corrcoef(a, b)
        return float(C[0, 1]) if C.shape == (2, 2) else float("nan")

    if pitch_idx is not None:
        mae_pitch = mae(pred_real[:, pitch_idx], Yv_real[:, pitch_idx])
        corr_pitch = corr(pred_real[:, pitch_idx], Yv_real[:, pitch_idx])
        print(f"[EVAL] MAE pitch_hz = {mae_pitch:.2f} Hz | corr = {corr_pitch:.3f}")

    if decay_idx is not None:
        mae_decay = mae(pred_real[:, decay_idx], Yv_real[:, decay_idx])
        corr_decay = corr(pred_real[:, decay_idx], Yv_real[:, decay_idx])
        print(f"[EVAL] MAE decay_t60 = {mae_decay:.3f} s | corr = {corr_decay:.3f}")

    overall_mae = {name: mae(pred_real[:, j], Yv_real[:, j]) for j, name in enumerate(names_order)}
    print("[EVAL] per-parameter MAE (real units):", {k: (round(v, 3)) for k, v in overall_mae.items()})

if __name__ == "__main__":
    main()
