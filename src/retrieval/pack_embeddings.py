# src/retrieval/pack_embeddings.py
import os, json, argparse, numpy as np
from glob import glob

def load_dir(dir_path, suffix):
    files = sorted(glob(os.path.join(dir_path, f"*.{suffix}.npy")))
    ids = [os.path.basename(f).replace(f".{suffix}.npy","") for f in files]
    arr = np.stack([np.load(f) for f in files])  # (N, D)
    return ids, arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir",  required=True)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    text_ids, text = load_dir(args.text_dir,  "text")
    audio_ids, audio = load_dir(args.audio_dir, "audio")

    if text.shape[0] != audio.shape[0] or text_ids != audio_ids:
        raise RuntimeError("Text/Audio mismatch: counts or ordering differ")

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
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[PACK] Saved: {args.out_dir} (N={len(ids)}, D_text={text.shape[1]}, D_audio={audio.shape[1]})")

if __name__ == "__main__":
    main()
