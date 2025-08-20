import os, json, csv, argparse, yaml
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embeddings.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    meta_csv = cfg["paths"]["meta_csv"]
    text_dir = Path(cfg["paths"]["text_emb_dir"])
    audio_dir = Path(cfg["paths"]["audio_emb_dir"])

    rows = list(csv.DictReader(open(meta_csv, "r", encoding="utf-8")))
    names = [Path(r["filename"]).stem for r in rows]

    # Counts
    text_files  = sorted([p for p in text_dir.glob("*.text.npy")])
    audio_files = sorted([p for p in audio_dir.glob("*.audio.npy")])

    print(f"[SMOKE] prompts: {len(names)} | text: {len(text_files)} | audio: {len(audio_files)}")

    # Basic dim check
    tdim = json.load(open(text_dir / "meta.json"))["dim"]
    adim = json.load(open(audio_dir / "meta.json"))["dim"]
    print(f"[SMOKE] dims: text={tdim}, audio={adim}")

    # Name coverage
    missing_text = [n for n in names if not (text_dir / f"{n}.text.npy").exists()]
    missing_audio = [n for n in names if not (audio_dir / f"{n}.audio.npy").exists()]

    assert not missing_text,  f"Missing text embeddings for: {missing_text[:5]} ..."
    assert not missing_audio, f"Missing audio embeddings for: {missing_audio[:5]} ..."
    print("[SMOKE] All embeddings present and consistent ")

if __name__ == "__main__":
    main()
