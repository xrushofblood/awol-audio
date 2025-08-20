import os, csv, json, argparse, yaml
from pathlib import Path
import numpy as np
import torch

def device_from_cfg(cfg):
    import torch
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def load_text_model(cfg, device):
    name = cfg["model"]["name"]
    if name == "laion_clap":
        import os, laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        # Robust checkpoint loading:
        ckpt_path = os.environ.get("CLAP_CKPT_PATH", "").strip()
        try:
            if ckpt_path:
                print(f"[TEXT] Loading CLAP checkpoint from CLAP_CKPT_PATH={ckpt_path}")
                model.load_ckpt(ckpt_path)
            else:
                print("[TEXT] Loading CLAP default checkpoint (no arg)…")
                model.load_ckpt()  # <-- default pretrained (avoids treating '2023' as file path)
        except Exception as e:
            print(f"[TEXT][WARN] CLAP checkpoint load failed: {e}")
            print("[TEXT][WARN] Falling back to sentence-transformers.")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(cfg["model"].get("sbert_name", "sentence-transformers/all-MiniLM-L6-v2"),
                                       device=device)
            return ("sentence_transformers", model)
        return ("laion_clap", model)
    elif name == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(cfg["model"]["sbert_name"], device=device)
        return ("sentence_transformers", model)
    else:
        raise ValueError(f"Unknown text model: {name}")

def encode_texts(model_tuple, texts):
    name, model = model_tuple
    if name == "laion_clap":
        with torch.no_grad():
            embs = model.get_text_embedding(texts)
        return np.asarray(embs, dtype=np.float32)
    else:
        embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embeddings.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    device = device_from_cfg(cfg)
    meta_csv = Path(cfg["paths"]["meta_csv"])
    out_dir = Path(cfg["paths"]["text_emb_dir"])
    ensure_dir(out_dir)

    model_tuple = load_text_model(cfg, device)
    print(f"[TEXT] Using {model_tuple[0]} on device={device}")

    rows = list(csv.DictReader(open(meta_csv, "r", encoding="utf-8")))
    texts = [r["prompt"] for r in rows]
    names = [Path(r["filename"]).stem for r in rows]

    embs = encode_texts(model_tuple, texts)  # (N, D)
    assert embs.shape[0] == len(names), "Embedding count mismatch."

    for name, vec in zip(names, embs):
        np.save(out_dir / f"{name}.text.npy", vec.astype(np.float32))

    meta = {"model": model_tuple[0], "dim": int(embs.shape[1]), "count": int(embs.shape[0])}
    json.dump(meta, open(out_dir / "meta.json", "w"))
    print(f"[TEXT] Saved {meta['count']} embeddings to {out_dir}")

if __name__ == "__main__":
    main()
