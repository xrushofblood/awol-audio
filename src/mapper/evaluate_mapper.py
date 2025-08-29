# src/mapper/evaluate_mapper.py
import argparse, yaml, json, math
from pathlib import Path
import numpy as np
import torch
import faiss
from sklearn.model_selection import train_test_split

from src.mapper.mapper_mlp import MLPMapper
from src.synth.param_regressor import ParamRegressor
from src.synth.targets_from_npz import targets_from_npz

EPS = 1e-9

# ---------- utils ----------
def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + EPS
    return x / n

def load_names(text_dir, audio_dir):
    text = {p.stem.replace(".text","") for p in Path(text_dir).glob("*.text.npy")}
    audio = {p.stem.replace(".audio","") for p in Path(audio_dir).glob("*.audio.npy")}
    common = sorted(list(text & audio))
    if not common:
        raise RuntimeError("No paired embeddings (.text.npy & .audio.npy) found.")
    return common

def load_matrix(dir_path: str, names: list, suffix: str) -> np.ndarray:
    X = [np.load(Path(dir_path) / f"{n}.{suffix}.npy").astype("float32") for n in names]
    return l2norm(np.stack(X, axis=0)).astype("float32")

def to_unit(x, spec):
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
    lo, hi = float(spec["min"]), float(spec["max"])
    scale = spec.get("scale", "linear")
    u = np.clip(np.asarray(u, dtype=np.float32), 0.0, 1.0)
    if scale == "log_hz":
        lo_log, hi_log = math.log2(lo), math.log2(hi)
        x_log = lo_log + u * (hi_log - lo_log)
        return (2.0 ** x_log).astype(np.float32)
    else:
        return (lo + u * (hi - lo)).astype(np.float32)

def load_targets01(npz_dir: str, names: list, audio_cfg: dict, specs: list) -> np.ndarray:
    Y_real = []
    for n in names:
        d = np.load(Path(npz_dir) / f"{n}.npz")
        y = targets_from_npz(
            d,
            sr=audio_cfg["sample_rate"],
            hop=audio_cfg["hop_size"],
            fmin_hz=float(audio_cfg.get("fmin_hz", 30.0)),
            fmax_hz=float(audio_cfg.get("fmax_hz", 12000.0)),
        )
        Y_real.append(y)
    Y_real = np.stack(Y_real, axis=0).astype("float32")
    Y01 = np.zeros_like(Y_real, dtype=np.float32)
    for j, spec in enumerate(specs):
        Y01[:, j] = np.clip(to_unit(Y_real[:, j], spec), 0.0, 1.0)
    return Y01

def load_faiss(index_dir: str):
    idx_p = Path(index_dir) / "faiss_ip.index"
    nms_p = Path(index_dir) / "names.npy"
    if not idx_p.exists() or not nms_p.exists():
        return None, None
    index = faiss.read_index(str(idx_p))
    names = np.load(nms_p, allow_pickle=True).tolist()
    return index, names

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper_real.yaml")
    ap.add_argument("--ckpt",   default="checkpoints/mapper_real/mapper_best.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    text_dir   = cfg["paths"]["text_emb_dir"]
    audio_dir  = cfg["paths"]["audio_emb_dir"]
    npz_dir    = cfg["paths"]["npz_dir"]
    index_dir  = cfg["paths"]["index_dir"]
    param_ckpt = cfg["paths"]["paramreg_ckpt"]
    params_cfg_path = cfg["paths"]["params_config"]

    # leggi specs + audio cfg dal params_config (quello usato per il paramreg)
    params_cfg = yaml.safe_load(open(params_cfg_path, "r"))
    specs = params_cfg["synth"]["params"]
    audio_cfg = params_cfg["audio"]
    param_names = [p["name"] for p in specs]

    # split come nel training (val set)
    all_names = load_names(text_dir, audio_dir)
    _, va_names = train_test_split(
        all_names, test_size=cfg["train"]["val_ratio"],
        random_state=cfg["train"]["seed"], shuffle=True
    )
    names = va_names

    # dati
    T = load_matrix(text_dir,  names, "text")   # (N,512)
    A = load_matrix(audio_dir, names, "audio")  # (N,512)
    Y01 = load_targets01(npz_dir, names, audio_cfg, specs)  # (N,P)

    # mapper
    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"],
        hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"],
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()

    # pred embedding audio
    with torch.no_grad():
        P = model(torch.from_numpy(T)).numpy()  # (N,512)
    P = l2norm(P).astype("float32")

    # cosine vs embedding audio reale
    cos = np.sum(P * A, axis=1)
    print(f"[EVAL] mean cosine similarity (val): {float(np.mean(cos)):.4f}")

    # retrieval (se indice presente)
    index, index_names = load_faiss(index_dir)
    if index is not None and index.ntotal > 0:
        maxk = 10
        D, I = index.search(P, maxk)
        r_at = {1:0, 5:0, 10:0}
        total = len(names)
        for qi, gold in enumerate(names):
            preds = [index_names[j] for j in I[qi]]
            if gold in preds[:1]:  r_at[1]  += 1
            if gold in preds[:5]:  r_at[5]  += 1
            if gold in preds[:10]: r_at[10] += 1
        print(f"[EVAL] R@1:  {r_at[1]/total:.3f}  (N={total})")
        print(f"[EVAL] R@5:  {r_at[5]/total:.3f}  (N={total})")
        print(f"[EVAL] R@10: {r_at[10]/total:.3f} (N={total})")
    else:
        print("[EVAL] Retrieval: index non trovato o vuoto — salto R@k.")

    # paramreg congelato -> predizione parametri (in [0,1])
    p_ckpt = torch.load(param_ckpt, map_location="cpu")
    p_cfg  = p_ckpt.get("cfg", None)
    if p_cfg is None:
        in_dim = 512; out_dim = len(specs); hidden = [256,128]; dropout = 0.0
    else:
        in_dim  = p_cfg["model"]["in_dim"]
        out_dim = p_cfg["model"]["out_dim"]
        hidden  = p_cfg["model"]["hidden"]
        dropout = p_cfg["model"]["dropout"]

    paramreg = ParamRegressor(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)
    paramreg.load_state_dict(p_ckpt["state_dict"] if "state_dict" in p_ckpt else p_ckpt)
    paramreg.eval()

    with torch.no_grad():
        pred01 = paramreg(torch.from_numpy(P)).numpy()
    pred01 = np.clip(pred01, 0.0, 1.0)

    # MAE per-parameter in unità reali
    mae_real = {}
    for j, spec in enumerate(specs):
        y_real    = from_unit(Y01[:, j], spec)
        pred_real = from_unit(pred01[:, j], spec)
        mae_real[param_names[j]] = float(np.mean(np.abs(pred_real - y_real)))

    # stampa rapida per pitch/decay
    if "pitch_hz" in mae_real:
        print(f"[EVAL] MAE pitch_hz: {mae_real['pitch_hz']:.3f} Hz")
    if "decay_t60" in mae_real:
        print(f"[EVAL] MAE decay_t60: {mae_real['decay_t60']:.3f} s")

    print("[EVAL] per-parameter MAE (real units):", {k: round(v, 4) for k, v in mae_real.items()})

if __name__ == "__main__":
    main()
