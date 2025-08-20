# src/mapper/predict_and_retrieve.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, yaml, json
from pathlib import Path
import numpy as np
import torch
import faiss

from .mapper_mlp import MLPMapper

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n

def load_faiss(index_dir):
    index = faiss.read_index(str(Path(index_dir) / "faiss_ip.index"))
    names = np.load(Path(index_dir) / "names.npy", allow_pickle=True).tolist()
    return index, names

def device_from_cfg(cfg):
    import torch
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    ap.add_argument("--ckpt", default="checkpoints/mapper/mapper_best.pt")
    ap.add_argument("--name", required=True, help="filename stem to use its saved text embedding, e.g., synthetic_pluck_012")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)

    text_dir = Path(cfg["paths"]["text_emb_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])

    # Load text embedding by filename stem:
    tpath = text_dir / f"{args.name}.text.npy"
    assert tpath.exists(), f"Missing text embedding: {tpath}"
    t = np.load(tpath).astype(np.float32)
    t = t / (np.linalg.norm(t) + 1e-9)
    t = torch.from_numpy(t).unsqueeze(0).to(device)

    # Load model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"], hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"]
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    with torch.no_grad():
        p = model(t).cpu().numpy()
    p = l2norm(p).astype(np.float32)

    index, names = load_faiss(index_dir)
    D, I = index.search(p, args.topk)
    D, I = D[0], I[0]

    print("\n[PRED→RETR RESULTS]")
    for rank, (idx, score) in enumerate(zip(I, D), 1):
        print(f"{rank:2d}. {names[idx]}  (score={score:.4f})")

if __name__ == "__main__":
    main()
