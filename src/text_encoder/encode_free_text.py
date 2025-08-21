# src/text_encoder/encode_free_text.py
import os
import argparse
import numpy as np

# Workaround Windows OpenMP duplicate (avoids crash with faiss/torch/numba)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_CLAP_CACHE = {"model": None, "variant": None}

def _load_clap(variant: str = "checkpoints/clap/clap_630k.pt"):
    import torch
    import laion_clap
    global _CLAP_CACHE
    if _CLAP_CACHE["model"] is not None and _CLAP_CACHE["variant"] == variant:
        return _CLAP_CACHE["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    
    model.load_ckpt(variant)
    _CLAP_CACHE["model"] = model
    _CLAP_CACHE["variant"] = variant
    return model

def encode_free_text(prompt: str, variant: str = "checkpoints/clap/clap_630k.pt") -> np.ndarray:
    import torch
    model = _load_clap(variant)
    with torch.no_grad():
        emb = model.get_text_embedding([prompt])
    emb = np.asarray(emb, dtype=np.float32)  # (1, D)
    # L2-norm a riga
    n = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="Free text prompt")
    ap.add_argument("--variant", default="checkpoints/clap/clap_630k.pt", help="Checkpoint CLAP (eg. 'clap-630k')")
    ap.add_argument("--save", default=None, help="Optional path to save .npy")
    args = ap.parse_args()

    vec = encode_free_text(args.prompt, args.variant)
    checksum = float(np.sum(vec))  # fast debug 
    print(f"[ENC] dim={vec.shape[1]} | checksum={checksum:.6f}")
    if args.save:
        np.save(args.save, vec.astype(np.float32))
        print(f"[ENC] saved to {args.save}")

if __name__ == "__main__":
    main()
