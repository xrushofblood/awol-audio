import os, json, argparse, yaml
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch

TARGET_SR = 48000  # CLAP expects 48 kHz internally; resample for robustness.

def device_from_cfg(cfg):
    d = str(cfg.get("device", "auto")).lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_audio_model(cfg, device):
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    # Use default checkpoint (no arg) to avoid treating a string as a local path.
    print("[AUDIO] Loading CLAP default checkpoint (no arg)…")
    model.load_ckpt()
    print(f"[AUDIO] Using LAION-CLAP on device={device}")
    return model

def load_audio_mono(fp: Path):
    y, sr = sf.read(fp, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    # Make sure it's float32 and contiguous
    y = np.asarray(y, dtype=np.float32)
    return y, int(sr)

def to_tensor_1xT(y: np.ndarray, sr: int, device: str):
    """Return a (1, T) float32 torch.Tensor on the requested device, resampled to TARGET_SR."""
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    # Optionally peak-normalize to avoid clipping extremes (kept mild)
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 1.0:
        y = y / peak
    # (T,) -> (1, T)
    y_t = torch.from_numpy(np.ascontiguousarray(y)).to(device=device, dtype=torch.float32).unsqueeze(0)
    return y_t, sr

def clap_embed_from_array(model, y: np.ndarray, sr: int, device: str):
    """
    Compute audio embedding for a mono numpy array using CLAP.
    Handles API variants by trying safe call signatures with a torch.Tensor.
    """
    y_t, sr = to_tensor_1xT(y, sr, device)

    with torch.no_grad():
        # Try positional signatures first (most common):
        #   get_audio_embedding_from_data(tensor, sr:int)
        try:
            emb = model.get_audio_embedding_from_data(y_t, sr)
        except TypeError:
            # Some builds assume 48k and take only the waveform tensor.
            try:
                emb = model.get_audio_embedding_from_data(y_t)
            except TypeError:
                # Very old forks might accept kwargs — last resort.
                try:
                    emb = model.get_audio_embedding_from_data(x=y_t, sr=sr)
                except TypeError as e:
                    raise RuntimeError(f"Unsupported CLAP API signature: {e}")

    # Ensure (D,) numpy float32
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embeddings.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    device = device_from_cfg(cfg)
    wav_dir = Path(cfg["paths"]["processed_wav_dir"])
    out_dir = Path(cfg["paths"]["audio_emb_dir"])
    ensure_dir(out_dir)

    model = load_audio_model(cfg, device)

    wavs = sorted([f for f in os.listdir(wav_dir) if f.lower().endswith(".wav")])
    if not wavs:
        raise RuntimeError(f"No .wav files found in {wav_dir}")

    all_embs = []
    for i, w in enumerate(wavs, 1):
        y, sr = load_audio_mono(wav_dir / w)
        emb = clap_embed_from_array(model, y, sr, device)
        np.save(out_dir / (Path(w).stem + ".audio.npy"), emb.astype(np.float32))
        all_embs.append(emb)
        if i % 10 == 0 or i == len(wavs):
            print(f"[AUDIO] {i}/{len(wavs)} done")

    dim = int(all_embs[0].shape[0]) if all_embs else 0
    meta = {"model": "laion_clap", "dim": dim, "count": len(all_embs), "sr": TARGET_SR}
    json.dump(meta, open(out_dir / "meta.json", "w"))
    print(f"[AUDIO] Saved {len(all_embs)} embeddings to {out_dir}")

if __name__ == "__main__":
    main()
