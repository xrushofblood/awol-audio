import os
import csv
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf

# Simple synthetic plucked-tone generator:
# - Exponentially decaying sinusoid with a few harmonics
# - Short noise burst at onset to mimic a pick attack
# - Variable brightness, decay, and pitch
#
# Output:
#   - WAV files in data/raw/
#   - meta/prompts.csv with (filename, prompt)

SR = 24000

def db_to_lin(db):
    return 10 ** (db / 20.0)

def normalize_peak(y, target_db=-1.0, eps=1e-9):
    peak = np.max(np.abs(y)) + eps
    gain = db_to_lin(target_db) / peak
    return (y * gain).astype(np.float32)

def synth_pluck(
    freq_hz=220.0,
    duration_s=1.0,
    decay_s=0.4,
    brightness=0.5,
    noise_attack_ms=10.0,
    seed=None
):
    """
    Generate a simple synthetic plucked tone.
    brightness in [0,1]: controls harmonic amplitudes and noise attack level.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = int(duration_s * SR)
    t = np.arange(n) / SR

    # Envelope: exponential decay
    env = np.exp(-t / max(decay_s, 1e-3))

    # Harmonic stack (up to 5 harmonics with roll-off controlled by brightness)
    num_harm = 5
    y = np.zeros_like(t)
    # rolloff: higher brightness => slower rolloff
    rolloff = np.interp(brightness, [0.0, 1.0], [0.8, 0.2])
    for h in range(1, num_harm + 1):
        amp = (1.0 / (h ** (1.0 + rolloff)))  # simple harmonic amplitude law
        y += amp * np.sin(2 * np.pi * (freq_hz * h) * t)

    # Short noise burst at onset (scaled by brightness)
    noise_len = int((noise_attack_ms / 1000.0) * SR)
    noise = (np.random.randn(noise_len) * np.interp(brightness, [0, 1], [0.05, 0.25])).astype(np.float32)
    y[:noise_len] += noise

    # Apply envelope
    y = (y * env).astype(np.float32)

    # Gentle highpass to reduce DC (one-pole)
    hp_alpha = 0.995
    y_hp = np.zeros_like(y)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(len(y)):
        x = y[i]
        y_hp[i] = hp_alpha * (prev_y + x - prev_x)
        prev_x = x
        prev_y = y_hp[i]

    # Peak normalize
    y_out = normalize_peak(y_hp, target_db=-1.0)
    return y_out

def describe_clip(freq_hz, decay_s, brightness):
    # Build a compact, human-readable prompt
    pitch_desc = "low" if freq_hz < 150 else "mid" if freq_hz < 400 else "high"
    decay_desc = "short decay" if decay_s < 0.3 else "medium decay" if decay_s < 0.6 else "long decay"
    bright_desc = "dark" if brightness < 0.33 else "balanced" if brightness < 0.66 else "bright"
    return f"{bright_desc} plucked tone, {pitch_desc} pitch, {decay_desc}"

def main():
    root = Path(__file__).resolve().parents[2]  # repo root
    raw_dir = root / "data" / "raw"
    meta_dir = root / "data" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Define a small grid of parameters
    freqs = [110.0, 146.8, 196.0, 261.6, 329.6, 392.0]  # A2, D3, G3, C4, E4, G4
    decays = [0.2, 0.4, 0.7]
    brights = [0.2, 0.5, 0.8]

    rows = []
    idx = 1
    for f in freqs:
        for d in decays:
            for b in brights:
                y = synth_pluck(freq_hz=f, duration_s=1.0, decay_s=d, brightness=b, noise_attack_ms=8.0)
                fname = f"synthetic_pluck_{idx:03d}.wav"
                sf.write(raw_dir / fname, y, SR)
                prompt = describe_clip(f, d, b)
                rows.append({"filename": fname, "prompt": prompt})
                idx += 1

    # Write prompts.csv
    csv_path = meta_dir / "prompts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prompt"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Generated {len(rows)} WAV files in {raw_dir}")
    print(f"Wrote metadata CSV at {csv_path}")

if __name__ == "__main__":
    main()