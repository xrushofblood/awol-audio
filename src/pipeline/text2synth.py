# -*- coding: utf-8 -*-
"""
Text → (CLAP text emb) → Mapper → (audio emb) → ParamRegressor → params → (optional) synth KS → .wav
- Selects the text encoder from config: “clap” (512-D, recommended) or “sbert” (384-D).
- Checks the consistency of the dimensions with the mapper (in_dim).
- Optional: performs top-k retrieval if it finds a FAISS index in paths.index_dir (faiss_ip.index + names.npy).
- Saves the predicted parameters in JSON and, if --no_render is not passed, renders a .wav via a simple internal Karplus-Strong.
Requirements:
  pip install laion-clap sentence-transformers faiss-cpu soundfile
"""

import os
import re
import json
import math
import argparse
import yaml
import numpy as np
from pathlib import Path

import torch
import soundfile as sf

# Mapper & ParamRegressor (tuoi moduli)
from src.mapper.mapper_mlp import MLPMapper
from src.synth.param_regressor import ParamRegressor

# --- utils --------------------------------------------------------------------

EPS = 1e-9

def l2norm_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + EPS
    return x / n

def l2norm_torch(x: torch.Tensor):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)

def device_from_cfg(cfg):
    d = (cfg.get("device") or "auto").lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def safe_stem(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_\- ]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s[:64] if len(s) > 64 else s

# --- text embedding -----------------------------------------------------------

def get_text_embedding(prompt: str, cfg: dict, device: str) -> torch.Tensor:
    """
    Returns a (1, D_text) torch.FloatTensor on `device`.
    Chooses CLAP (512-D) or SBERT (384-D) depending on cfg["text_encoder"].
    """
    enc = (cfg.get("text_encoder") or "clap").lower()
    if enc == "clap":
        # LAION-CLAP text encoder
        from laion_clap import CLAP_Module
        clap_ckpt = cfg["model"]["laion_variant"]
        clap = CLAP_Module(enable_fusion=False, device=device)
        clap.load_ckpt(clap_ckpt)
        with torch.no_grad():
            e = clap.get_text_embedding([prompt])  # (1,512), numpy
        e = e.astype(np.float32)[0]
    else:
        # SBERT encoder (384-D)
        from sentence_transformers import SentenceTransformer
        sbert_name = cfg["model"]["sbert_name"]
        sbert = SentenceTransformer(sbert_name, device=device if device == "cuda" else None)
        e = sbert.encode(prompt, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)

    e = l2norm_np(e)
    return torch.from_numpy(e).unsqueeze(0).to(device)  # (1, D)

# --- FAISS retrieval (optional) ---------------------------------------------

# --- Retrieval (optional) ---
def try_retrieval(index_dir: Path, qvec: np.ndarray, topk: int):
    """
    Return [] if topk<=0 or index files missing.
    Otherwise, search FAISS and return list of (name, score).
    """
    if topk is None or topk <= 0:
        return []

    idx_path = Path(index_dir) / "faiss_ip.index"
    names_path = Path(index_dir) / "names.npy"
    if not idx_path.exists() or not names_path.exists():
        print(f"[RET] index not found in {index_dir} -> skipping retrieval.")
        return []

    import faiss
    index = faiss.read_index(str(idx_path))
    names = np.load(names_path, allow_pickle=True).tolist()

    q = qvec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    D, I = index.search(q, topk)  # safe: topk>0
    D, I = D[0], I[0]

    hits = []
    for score, idx in zip(D, I):
        if 0 <= idx < len(names):
            hits.append((names[idx], float(score)))
    return hits

# --- (de)normalization of params (matches your params_real.yaml spec) --------

def to_unit(x, spec):
    """Map real-valued target -> [0,1] with optional 'scale: log_hz' for pitch."""
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
    """Map normalized [0,1] -> real-valued with optional 'scale: log_hz'."""
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

# --- minimal Karplus-Strong renderer (self-contained) ------------------------

def render_karplus_strong(params: dict, sr: int = 44100) -> np.ndarray:
    """
    Very small KS-style pluck to let you *hear* something end-to-end.
    Not physically exact, but deterministic and fast.

    Uses:
      - pitch_hz        -> base frequency
      - decay_t60 (s)   -> sets loop damping
      - brightness      -> simple one-pole tone control on the loop
      - noise_mix       -> amount of noise in the initial burst
      - pick_position   -> shapes the initial burst (rough comb)
    """
    f0 = float(max(30.0, params.get("pitch_hz", 110.0)))
    t60 = float(max(0.05, params.get("decay_t60", 0.3)))
    bright = float(np.clip(params.get("brightness", 0.2), 0.0, 1.0))
    noise_mix = float(np.clip(params.get("noise_mix", 0.1), 0.0, 1.0))
    pick_pos = float(np.clip(params.get("pick_position", 0.5), 0.1, 0.9))

    # make duration proportional to t60 (cap reasonable)
    dur = float(np.clip(t60 * 1.2, 0.3, 2.5))
    N = int(sr * dur)
    if N < 2:
        N = 2

    # delay line length
    L = max(2, int(round(sr / f0)))
    buf = np.random.randn(L).astype(np.float32)

    # initial excitation: mix noise + single-cycle sine (to stabilize pitch)
    t = np.arange(L, dtype=np.float32) / sr
    sine = np.sin(2 * np.pi * f0 * t).astype(np.float32)
    buf = noise_mix * buf + (1.0 - noise_mix) * sine

    # crude pick-position comb (zero at pick_pos of string)
    # apply simple comb by delaying and subtracting a small portion
    d_samp = int(np.clip(int(L * pick_pos), 1, L - 1))
    comb = np.zeros_like(buf)
    comb[d_samp:] = buf[:-d_samp]
    buf = buf - 0.5 * comb  # mild notch

    # loop filter:
    #   - set feedback to meet T60: |g|^N ≈ 1e-3  → g ≈ 10^(-3/N)
    #   - simple one-pole lowpass to control "brightness"
    g = 10.0 ** (-3.0 / (t60 * f0))  # empirical
    g = float(np.clip(g, 0.9, 0.9999))

    # one-pole coeff for brightness; low bright -> stronger LP (darker)
    # map bright [0..1] -> alpha in [0.2..0.95]
    alpha = 0.2 + 0.75 * float(bright)
    y = np.zeros(N, dtype=np.float32)

    lp_state = 0.0
    idx = 0
    for n in range(N):
        xn = buf[idx]
        # simple low-pass in the feedback path
        lp_state = alpha * xn + (1 - alpha) * lp_state
        yn = lp_state
        y[n] = yn
        # update delay line (KS average with damping)
        nxt = g * 0.5 * (buf[idx] + buf[(idx + 1) % L])
        buf[idx] = nxt
        idx = (idx + 1) % L

    # mild output normalization
    peak = float(np.max(np.abs(y)) + 1e-9)
    y = 0.95 * y / peak
    return y

# --- main pipeline ------------------------------------------------------------

def load_mapper(mapper_cfg_path: Path, mapper_ckpt_path: Path, device: str) -> MLPMapper:
    mcfg = yaml.safe_load(open(mapper_cfg_path, "r"))
    model = MLPMapper(
        in_dim=mcfg["model"]["in_dim"],
        hidden=tuple(mcfg["model"]["hidden"]),
        out_dim=mcfg["model"]["out_dim"],
        dropout=mcfg["model"]["dropout"],
        norm_out=mcfg["model"].get("norm_out", True),
    ).to(device)
    state = torch.load(mapper_ckpt_path, map_location="cpu")
    state_dict = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state_dict)
    model.eval()
    return model, mcfg

def load_paramreg(params_cfg_path: Path, param_ckpt_path: Path, device: str):
    pcfg = yaml.safe_load(open(params_cfg_path, "r"))
    model = ParamRegressor(
        in_dim=pcfg["model"]["in_dim"],
        hidden=pcfg["model"]["hidden"],
        out_dim=pcfg["model"]["out_dim"],
        dropout=pcfg["model"]["dropout"],
    ).to(device)
    state = torch.load(param_ckpt_path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict)
    model.eval()
    return model, pcfg

def predict_audio_embed(mapper: MLPMapper, t_emb: torch.Tensor, device: str) -> np.ndarray:
    """Returns (1, D) np.float32, L2-normalized."""
    with torch.no_grad():
        p = mapper(t_emb.to(device))
        p = l2norm_torch(p).cpu().numpy().astype("float32")
    return p

def predict_params_from_embed(param_model: ParamRegressor, params_cfg: dict, a_emb: np.ndarray, device: str) -> dict:
    """a_emb: (1,512) np.float32. Returns dict name->value (real units)."""
    specs = params_cfg["synth"]["params"]
    with torch.no_grad():
        y01 = param_model(torch.from_numpy(a_emb).to(device)).cpu().numpy()[0]  # [0,1]
    cols = []
    names = []
    for j, spec in enumerate(specs):
        names.append(spec["name"])
        cols.append(from_unit(y01[j], spec))
    vals = np.stack(cols).astype(float).tolist()
    return {n: v for n, v in zip(names, vals)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline config (YAML)")
    ap.add_argument("--query", required=True, help="text prompt")
    ap.add_argument("--topk", type=int, default=5, help="optional retrieval top-k")
    ap.add_argument("--no_render", action="store_true", help="skip audio rendering")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)

    # load models
    mapper, mapper_cfg = load_mapper(Path(cfg["paths"]["mapper_config"]),
                                     Path(cfg["paths"]["mapper_ckpt"]),
                                     device)
    param_model, params_cfg = load_paramreg(Path(cfg["paths"]["params_config"]),
                                            Path(cfg["paths"]["param_ckpt"]),
                                            device)

    # text embedding
    t = get_text_embedding(args.query, cfg, device)  # (1, D_text)

    # sanity check: text dim == mapper in_dim
    d_text = t.shape[-1]
    expected = int(mapper_cfg["model"]["in_dim"])
    assert d_text == expected, (
        f"Text embedding dim {d_text} != mapper in_dim {expected}. "
        f"Set text_encoder=clap in config (or retrain a mapper with in_dim={d_text})."
    )

    # predict audio embedding
    a_pred = predict_audio_embed(mapper, t, device)  # (1, D)
    print("[OK] predicted audio embedding shape:", a_pred.shape)

    # optional retrieval
    idx_dir = Path(cfg["paths"].get("index_dir", ""))
    if idx_dir and idx_dir.exists():
        hits = try_retrieval(idx_dir, a_pred, topk=args.topk)
        if hits:
            print("\n[RETRIEVAL top-{}]".format(args.topk))
            for r, (name, score) in enumerate(hits, 1):
                print(f" {r:2d}. {name}  (score={score:.4f})")

    # predict synth parameters
    params = predict_params_from_embed(param_model, params_cfg, a_pred, device)
    print("\n[PREDICTED PARAMS]")
    for k, v in params.items():
        if isinstance(v, (list, tuple, np.ndarray)):
            print(f" - {k}: {v}")
        else:
            print(f" - {k}: {float(v):.6f}")

    # output paths
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(args.query)
    json_path = out_dir / f"{stem}.params.json"
    wav_path  = out_dir / f"{stem}.wav"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"\n[SAVED] params → {json_path}")

    # render (optional)
    if not args.no_render:
        sr = int(params_cfg["audio"].get("sample_rate", 44100)) if "audio" in params_cfg else 44100
        y = render_karplus_strong(params, sr=sr)
        sf.write(str(wav_path), y, sr)
        print(f"[SAVED] audio  → {wav_path}")
    else:
        print("[SKIP] rendering disabled (--no_render)")

if __name__ == "__main__":
    main()