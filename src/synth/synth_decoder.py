# src/synth/synth_decoder.py
from __future__ import annotations
import math
from typing import Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf


# ----------------------
# Utility: save wav file
# ----------------------
def save_wave(wave: np.ndarray, sr: int, path: str) -> None:
    """Save mono waveform with gentle limiting to avoid blasts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wave = np.clip(wave.astype(np.float32), -0.95, 0.95)
    sf.write(path, wave, sr)


# ----------------------
# Simple MLP decoder (reference)
# ----------------------
class SimpleDecoder(nn.Module):
    """
    Very simple MLP "decoder" that maps an embedding to a waveform directly.
    Kept for reference; not recommended for Day 6 because variety was poor.
    """
    def __init__(self, input_dim: int = 512, hidden: Tuple[int, ...] = (512, 512), out_samples: int = 16000):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, out_samples), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, D)
        returns: (B, T) in [-1, 1]
        """
        return self.net(emb)


# --------------------------------------
# Karplus–Strong Decoder (Day 6 version)
# --------------------------------------
class KarplusStrongDecoder(nn.Module):
    """
    Karplus–Strong plucked-string style decoder.
    We map the embedding to audible controls:
      - f0 (Hz)
      - decay time (seconds) -> per-sample feedback / damping
      - brightness (0..1) -> low-pass shaping in feedback loop
      - excitation color (0..1) -> mixes white noise and short burst
    """

    def __init__(self, sample_rate: int = 16000, seed: int | None = None):
        super().__init__()
        self.sr = int(sample_rate)
        self.rng = np.random.RandomState(seed if seed is not None else 0)

    @staticmethod
    def _squash_01(x: torch.Tensor) -> torch.Tensor:
        # Map roughly [-1, 1] -> [0, 1]; works fine for L2-normalized embeddings
        return (x + 1.0) * 0.5

    def emb_to_params(self, emb: torch.Tensor) -> dict:
        """
        Map embedding (B, D) to param dicts of shape (B,).
        We only use first few dimensions; you can re-map later as you wish.
        """
        assert emb.ndim == 2, "Expected emb of shape (B, D)"
        e0 = self._squash_01(emb[:, 0])  # pitch control
        e1 = self._squash_01(emb[:, 1])  # decay control
        e2 = self._squash_01(emb[:, 2])  # brightness control
        e3 = self._squash_01(emb[:, 3])  # excitation color

        # Pitch mapping: 120..880 Hz (roughly A2..A5)
        f0 = 120.0 + (880.0 - 120.0) * e0

        # Decay time: 0.06..0.60 s (audibly different on 1 s signals)
        decay_s = 0.06 + (0.60 - 0.06) * e1

        # Brightness: 0.2..0.98 for LP coefficient in feedback
        brightness = 0.20 + (0.98 - 0.20) * e2

        # Excitation mix: 0 (noisier/darker) .. 1 (brighter burst)
        excite = e3

        return {
            "f0_hz": f0,               # (B,)
            "decay_s": decay_s,        # (B,)
            "brightness": brightness,  # (B,)
            "excite": excite,          # (B,)
        }

    def _excitation(self, length: int, excite_mix: float) -> np.ndarray:
        """Create an excitation signal combining white noise and a short bright burst."""
        noise = self.rng.randn(length).astype(np.float32)
        burst_len = max(4, length // 16)
        burst = np.zeros(length, dtype=np.float32)
        burst[:burst_len] = 1.0
        ex = float(excite_mix) * burst + (1.0 - float(excite_mix)) * noise
        ex = ex / (np.max(np.abs(ex)) + 1e-6)
        return ex

    def _ks_string(self, f0: float, seconds: float, decay_s: float, brightness: float, excite_mix: float) -> np.ndarray:
        """Single-string Karplus–Strong with simple damping/brightness."""
        sr = self.sr
        N = int(sr * seconds)
        N = max(N, 1)

        # delay length
        delay = max(2, int(round(sr / max(40.0, float(f0)))))  # clamp to min freq ~40 Hz
        buf = self._excitation(delay, excite_mix)

        # Per-sample decay coefficient ~ exp(-1 / (tau * sr))
        tau = max(0.01, float(decay_s))
        decay_coeff = math.exp(-1.0 / (tau * sr))

        # Brightness as 1-pole LP in the feedback path: y[n] = b*y[n-1] + (1-b)*avg
        b = float(np.clip(brightness, 0.0, 0.999))
        y_prev = 0.0

        out = np.zeros(N, dtype=np.float32)
        idx = 0
        for n in range(N):
            x = buf[idx]
            avg = 0.5 * (x + y_prev)        # standard KS averaging
            y = b * y_prev + (1.0 - b) * avg  # brightness shaping
            out[n] = y
            y_prev = y
            buf[idx] = y * decay_coeff      # feedback with decay
            idx += 1
            if idx >= delay:
                idx = 0

        out /= (np.max(np.abs(out)) + 1e-6)
        out *= 0.9
        return out.astype(np.float32)

    def forward(self, emb: torch.Tensor, seconds: float = 1.0) -> torch.Tensor:
        """
        emb: (B, D)
        returns: (B, T) waveform in [-1, 1]
        """
        if emb.ndim != 2:
            raise ValueError("emb must be (B, D)")

        params = self.emb_to_params(emb)  # dict of tensors (B,)
        f0 = params["f0_hz"].cpu().numpy()
        decay_s = params["decay_s"].cpu().numpy()
        brightness = params["brightness"].cpu().numpy()
        excite = params["excite"].cpu().numpy()

        waves = []
        for i in range(emb.shape[0]):
            w = self._ks_string(
                f0=f0[i],
                seconds=float(seconds),
                decay_s=decay_s[i],
                brightness=brightness[i],
                excite_mix=excite[i],
            )
            waves.append(w)
        waves = np.stack(waves, axis=0)  # (B, T)
        return torch.from_numpy(waves)
