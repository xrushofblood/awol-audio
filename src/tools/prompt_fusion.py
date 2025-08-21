# src/tools/prompt_fusion.py
import os, argparse, yaml
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
def encode_free_text_list(prompts, variant="clap-630k") -> np.ndarray:
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
    emb = np.asarray(emb, dtype=np.float32)  # (N, D)
    emb = l2norm(emb)
    mean = np.mean(emb, axis=0, keepdims=True)
    return l2norm(mean)  # (1, D)

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    ap.add_argument("--ckpt", default="checkpoints/mapper/mapper_best.pt")
    ap.add_argument("--prompts", nargs="+", required=True, help='One of more phrases between quotation marks')
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)
    variant = args.variant or cfg.get("model", {}).get("laion_variant") or "checkpoints/clap/clap_630k.pt"

    # 1) media embedding of paraphrases
    t = encode_free_text_list(args.prompts, variant=variant)

    # 2) mapper
    model = load_mapper(cfg, Path(args.ckpt), device)
    with torch.no_grad():
        p = model(torch.from_numpy(t).to(device)).cpu().numpy()
    p = l2norm(p).astype(np.float32)

    # 3) retrieve
    index_dir = Path(cfg["paths"]["index_dir"])
    index, names = load_faiss(index_dir)
    D, I = index.search(p, args.topk)
    D, I = D[0], I[0]

    print("\n[PROMPT FUSION]")
    print("Prompts:", " | ".join(args.prompts))
    for rank, (idx, score) in enumerate(zip(I, D), 1):
        print(f"{rank:2d}. {names[idx]}  (score={score:.4f})")

if __name__ == "__main__":
    main()
