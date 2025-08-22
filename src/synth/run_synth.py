# src/synth/run_synth.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

from ..mapper.mapper_mlp import MLPMapper
from .synth_decoder import SimpleDecoder, KarplusStrongDecoder, save_wave


# --------- CLAP text encoding ----------
def encode_text_clap(prompt: str, ckpt_path: str, device: str = "cpu") -> np.ndarray:
    """
    Encode a free-text prompt into a 512-D CLAP text embedding.
    Returns L2-normalized numpy array of shape (1, 512).
    """
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt(ckpt_path)  # e.g. checkpoints/clap/clap_630k.pt
    with torch.no_grad():
        emb = model.get_text_embedding([prompt])  # (1, 512)
    emb = np.asarray(emb, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return emb


def choose_device(cfg_device: str) -> str:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth.yaml", help="Synthesis config file")
    ap.add_argument("--prompt", required=True, help="Free-text prompt to synthesize")
    ap.add_argument("--mapper_ckpt", default=None, help="(Optional) override path to mapper checkpoint")
    ap.add_argument("--clap_ckpt",   default=None, help="(Optional) override path to CLAP checkpoint")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for excitation")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = choose_device(cfg.get("device", "cpu"))

    # --------- Paths & settings ----------
    outdir      = Path(cfg["paths"]["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    mapper_ckpt = args.mapper_ckpt or cfg["paths"]["mapper_ckpt"]
    clap_ckpt   = args.clap_ckpt   or cfg["paths"]["clap_ckpt"]

    sr      = int(cfg["synth"]["sample_rate"])
    seconds = float(cfg["synth"]["seconds"])
    synth_type = cfg["synth"].get("type", "karplus_strong").lower()

    # --------- Encode prompt (CLAP) ----------
    t_emb = encode_text_clap(args.prompt, clap_ckpt, device=device)  # (1, 512)

    # --------- Load mapper (text → audio embedding) ----------
    in_dim = int(cfg["synth"].get("input_dim", 512))
    hidden = tuple(cfg["synth"].get("hidden", [512, 512]))
    mapper = MLPMapper(
        in_dim=in_dim, hidden=hidden, out_dim=in_dim,
        dropout=0.0, norm_out=True
    ).to(device)
    ckpt = torch.load(mapper_ckpt, map_location="cpu")
    mapper.load_state_dict(ckpt["model"])
    mapper.eval()

    # Predict audio embedding
    with torch.no_grad():
        t = torch.from_numpy(t_emb).to(device)      # (1, D)
        a_pred = mapper(t)
        a_pred = nn.functional.normalize(a_pred, dim=-1, eps=1e-8)  # (1, D)

    # --------- Choose decoder ----------
    if synth_type in ("karplus", "karplus_strong", "ks"):
        decoder = KarplusStrongDecoder(sample_rate=sr, max_seconds=seconds, seed=args.seed)
    elif synth_type in ("simple", "simple_decoder"):
        decoder = SimpleDecoder(input_dim=in_dim, hidden=hidden, sample_rate=sr, seconds=seconds)
    else:
        raise RuntimeError(f"Unknown synth.type '{synth_type}' in synth.yaml")

    decoder = decoder.to(device)
    decoder.eval()

    # Decode to waveform
    with torch.no_grad():
        wave = decoder(a_pred).cpu().numpy().reshape(-1)

    # Save WAV
    out_name = args.prompt.replace(" ", "_").replace("/", "-")
    out_path = outdir / f"{out_name}.wav"
    save_wave(wave, sr, str(out_path))
    print(f"[Day5] Saved audio to: {out_path}")

    # Debug stats
    rms = float(np.sqrt(np.mean(wave**2)) + 1e-12)
    print("[DBG] audio shape:", wave.shape,
          "min:", float(wave.min()),
          "max:", float(wave.max()),
          "rms:", rms)


if __name__ == "__main__":
    main()
