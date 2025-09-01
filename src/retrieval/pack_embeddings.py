# src/retrieval/pack_embeddings.py
import os, json, argparse, numpy as np
from glob import glob

def l2norm(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def load_dir(dir_path, suffix):
    files = sorted(glob(os.path.join(dir_path, f"*.{suffix}.npy")))
    if not files:
        raise RuntimeError(f"No *.{suffix}.npy found in {dir_path}")
    ids = [os.path.basename(f).replace(f".{suffix}.npy","") for f in files]
    arr = np.stack([np.load(f) for f in files])  # (N, D)
    return ids, arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir",  required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--no_norm",   action="store_true",
                    help="Do not apply L2-normalization (default: normalize).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text_ids, text = load_dir(args.text_dir,  "text")
    audio_ids, audio = load_dir(args.audio_dir, "audio")

    if text.shape[0] != audio.shape[0] or text_ids != audio_ids:
        raise RuntimeError("Text/Audio mismatch: counts or ordering differ")
    
    did_norm = not args.no_norm
    if did_norm:
        text  = l2norm(text)
        audio = l2norm(audio)

    ids = np.array(text_ids)
    np.save(os.path.join(args.out_dir, "text.npy"),  text)
    np.save(os.path.join(args.out_dir, "audio.npy"), audio)
    np.save(os.path.join(args.out_dir, "ids.npy"),   ids)

    meta = {
        "count": int(len(ids)),
        "dim_text": int(text.shape[1]),
        "dim_audio": int(audio.shape[1]),
        "source_text_dir": args.text_dir,
        "source_audio_dir": args.audio_dir,
        "l2_normalized": bool(did_norm)
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[PACK] Saved: {args.out_dir} (N={len(ids)}, D_text={text.shape[1]}, D_audio={audio.shape[1]})")

if __name__ == "__main__":
    main()
