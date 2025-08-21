# src/retrieval/batch_free_query.py
import os, argparse, yaml, csv
from pathlib import Path
import numpy as np
import torch
import faiss

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n

def load_faiss(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss_ip.index"))
    names = np.load(index_dir / "names.npy", allow_pickle=True).tolist()
    return index, names

def device_from_cfg(cfg):
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"

# CLAP cache
_CLAP_CACHE = {"model": None, "variant": None}
def encode_free_text_batch(prompts, variant="checkpoints/clap/clap_630k.pt") -> np.ndarray:
    import laion_clap
    global _CLAP_CACHE
    if _CLAP_CACHE["model"] is None or _CLAP_CACHE["variant"] != variant:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt(variant)
        _CLAP_CACHE["model"] = model
        _CLAP_CACHE["variant"] = variant
    with torch.no_grad():
        emb = _CLAP_CACHE["model"].get_text_embedding(prompts)
    emb = np.asarray(emb, dtype=np.float32)
    return l2norm(emb)

from ..mapper.mapper_mlp import MLPMapper

def load_mapper(cfg, ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"],
        hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

def _read_prompts(path: Path):
    if path.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [l for l in lines if l]
    elif path.suffix.lower() == ".csv":
        # usa la colonna 'prompt'
        rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
        return [r["prompt"].strip() for r in rows if r.get("prompt", "").strip()]
    else:
        raise ValueError("Input must be .txt (one query per line) or .csv with a ‘prompt’ column.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    ap.add_argument("--ckpt", default="checkpoints/mapper/mapper_best.pt")
    ap.add_argument("--input", required=True, help="File .txt o .csv con le query")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", required=True, help="CSV di output")
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)
    variant = args.variant or cfg.get("model", {}).get("laion_variant") or "checkpoints/clap/clap_630k.pt"

    prompts = _read_prompts(Path(args.input))
    if not prompts:
        print("[BATCH] No queries found.")
        return

    # 1) text -> emb CLAP (batch)
    T = encode_free_text_batch(prompts, variant=variant)  # (N, D)

    # 2) mapper (batch)
    model = load_mapper(cfg, Path(args.ckpt), device)
    with torch.no_grad():
        P = model(torch.from_numpy(T).to(device)).cpu().numpy()
    P = l2norm(P).astype(np.float32)

    # 3) retrieve
    index_dir = Path(cfg["paths"]["index_dir"])
    index, names = load_faiss(index_dir)
    D, I = index.search(P, args.topk)  # (N, K)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "rank", "name", "score"])
        for qi, q in enumerate(prompts):
            for r, (idx, sc) in enumerate(zip(I[qi], D[qi]), 1):
                w.writerow([q, r, names[idx], f"{float(sc):.6f}"])

    print(f"[BATCH] Saved results to {outp} (N={len(prompts)}, topk={args.topk})")

if __name__ == "__main__":
    main()
