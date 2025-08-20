# src/retrieval/retrieve.py
# Retrieval baseline over per-file CLAP audio embeddings.
# - Index is built from *.audio.npy under cfg["paths"]["audio_emb_dir"]
# - Query is encoded with CLAP text encoder on the chosen device
# - Cosine similarity via FAISS IndexFlatIP with L2-normalized vectors
import os
from pathlib import Path
import argparse
import yaml
import numpy as np
import faiss


def device_from_cfg(cfg) -> str:
    import torch
    d = str(cfg.get("device", "auto")).lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_clap_text_encoder(device: str):
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    # Default checkpoint (no arg) avoids treating strings as local paths
    print("[RETR] Loading CLAP default checkpoint (no arg)…")
    model.load_ckpt()
    return model


def encode_query_text(clap_model, query: str) -> np.ndarray:
    emb = clap_model.get_text_embedding([query])  # (1, D), torch tensor in most builds
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()
    else:
        emb = np.asarray(emb)
    emb = emb.astype(np.float32)[0]  # (D,)
    return emb


def l2norm(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    faiss.normalize_L2(X)
    return X


def build_index_from_dir(audio_emb_dir: Path, index_dir: Path):
    # Gather all per-file audio embeddings
    paths = sorted(audio_emb_dir.glob("*.audio.npy"))
    if not paths:
        raise FileNotFoundError(f"No *.audio.npy found in {audio_emb_dir}")

    X = np.stack([np.load(p).astype(np.float32) for p in paths])  # (N, D)
    X = l2norm(X)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    names = np.array([p.stem for p in paths], dtype=object)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss_ip.index"))
    np.save(index_dir / "names.npy", names)
    print(f"[RETR] Built FAISS index with {len(names)} items.")
    return index, names


def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss_ip.index"))
    names = np.load(index_dir / "names.npy", allow_pickle=True).tolist()
    return index, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embeddings.yaml")
    ap.add_argument("--query", required=True, type=str)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    audio_emb_dir = Path(cfg["paths"]["audio_emb_dir"])            # e.g., data/processed/embeddings/audio
    index_dir     = Path(cfg["paths"]["index_dir"])                # e.g., data/processed/index

    # Build or load index
    if args.rebuild or not (index_dir / "faiss_ip.index").exists():
        index, names = build_index_from_dir(audio_emb_dir, index_dir)
    else:
        index, names = load_index(index_dir)
        print(f"[RETR] Loaded FAISS index with {len(names)} items.")

    # Text encoder (CLAP)
    device = device_from_cfg(cfg)
    print(f"[RETR] Using device: {device}")
    clap = load_clap_text_encoder(device)

    # Encode query and search
    q = encode_query_text(clap, args.query).reshape(1, -1).astype(np.float32)
    q = l2norm(q)
    scores, idxs = index.search(q, args.topk)
    scores, idxs = scores[0], idxs[0]

    print("\n[RESULTS]")
    for rank, (i, sc) in enumerate(zip(idxs, scores), 1):
        print(f"{rank:2d}. {names[i]}  (score={sc:.4f})")


if __name__ == "__main__":
    main()
