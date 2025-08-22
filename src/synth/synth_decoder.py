# src/synth/synth_decoder.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf


# ---------- I/O ----------
def save_wave(wave: np.ndarray, sr: int, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # limiter soft: clip a ±0.95
    wave = np.clip(wave, -0.95, 0.95).astype(np.float32)
    sf.write(out_path, wave, sr)


# ---------- Simple MLP → waveform (baseline non-fisico) ----------
class SimpleDecoder(nn.Module):
    """
    Maps a 512-D (or cfg-defined) embedding to a raw waveform by a small MLP.
    This is a *placeholder* and not trained: useful come baseline per debug.
    """
    def __init__(self, input_dim: int = 512, hidden: Tuple[int, ...] = (512, 512),
                 sample_rate: int = 16000, seconds: float = 1.0):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.seconds = float(seconds)
        self.out_samples = int(self.sample_rate * self.seconds)

        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        layers += [nn.Linear(last, self.out_samples), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) normalized embedding
        returns: (B, T) waveform in [-1, 1]
        """
        return self.net(z)


# ---------- Karplus–Strong plucked string ----------
class KarplusStrongDecoder(nn.Module):
    """
    Differentiable-ish Karplus–Strong renderer.
    We *derive* physical-ish parameters (f0, decay, brightness, noise mix)
    FROM the input embedding with a tiny MLP, then synthesize the waveform.

    __init__(sample_rate, max_seconds, seed=None)
    forward(z) -> (B, T)
    """
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 1.0, seed: int | None = None):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.max_seconds = float(max_seconds)
        self.out_samples = int(self.sample_rate * self.max_seconds)

        # Tiny mapper: embedding D → params [f0_norm, decay, brightness, noise_mix]
        # NB: D viene dedotto a runtime dalla prima forward (lazy init) usando nn.LazyLinear.
        self.param_head = nn.Sequential(
            nn.LazyLinear(128), nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )

        # Range fisici (molto semplici)
        self.register_buffer("f0_min", torch.tensor(80.0))    # Hz
        self.register_buffer("f0_max", torch.tensor(1200.0))  # Hz

        # Random init for pluck noise
        self._rng = np.random.default_rng(seed if seed is not None else 0)

    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _params_from_embedding(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        z: (B, D) normalized embedding
        returns:
          f0_hz: (B,)
          decay: (B,)   in (0.85..0.999)
          bright: (B,)  in (0.0..1.0)
          mix: (B,)     in (0.0..1.0)  noise burst vs. sine excitation
        """
        raw = self.param_head(z)  # (B, 4)
        s = self._sigmoid(raw)

        f0_norm  = s[:, 0]                                   # (0..1)
        f0_hz    = self.f0_min + f0_norm * (self.f0_max - self.f0_min)

        decay    = 0.85 + 0.149 * s[:, 1]                    # (0.85..0.999)
        bright   = s[:, 2]                                   # (0..1)
        mix      = s[:, 3]                                   # (0..1)

        return f0_hz, decay, bright, mix

    def _ks_render_np(self, f0_hz: float, decay: float, bright: float, mix: float) -> np.ndarray:
        """
        Single-note Karplus–Strong in NumPy (CPU), 1 second max.
        - f0_hz: target pitch
        - decay: feedback gain per sample (close to 1.0 for long sustain)
        - bright: simple one-pole tone (0=darker, 1=brighter)
        - mix: excitation blend (0=pure burst noise, 1=pure sine)
        """
        T = self.out_samples
        sr = self.sample_rate

        # delay length in samples
        delay_samps = max(2, int(round(sr / max(20.0, f0_hz))))  # guard for numeric stability
        buf = np.zeros(delay_samps, dtype=np.float32)

        # excitation: burst noise + sine burst
        n_burst = min(delay_samps, max(8, int(0.01 * sr)))  # 10 ms burst cap
        noise_exc = (self._rng.random(n_burst) * 2.0 - 1.0).astype(np.float32)
        t = np.arange(n_burst, dtype=np.float32) / sr
        sine_exc = np.sin(2 * np.pi * f0_hz * t).astype(np.float32)
        exc = (1.0 - mix) * noise_exc + mix * sine_exc

        # write excitation in the delay buffer
        buf[:n_burst] = exc

        # One-pole lowpass in the feedback: y[n] = (1 - a)*x[n] + a*y[n-1]
        # Map brightness→a (0..1) where a near 0 = bright, a near 1 = dark
        a = float(1.0 - 0.9 * bright)  # clamp to (0.1..1.0)
        ylp = 0.0

        out = np.zeros(T, dtype=np.float32)
        idx = 0

        for n in range(T):
            # Read head
            yn = buf[idx]

            # Simple one-pole on feedback
            ylp = (1.0 - a) * yn + a * ylp
            fb = decay * ylp

            # Write head (wrap-around)
            buf[idx] = fb
            idx += 1
            if idx >= delay_samps:
                idx = 0

            out[n] = yn

        # light normalization
        peak = np.max(np.abs(out)) + 1e-7
        out = 0.95 * out / peak
        return out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) normalized embedding
        returns: (B, T) waveform in [-1, 1]
        """
        if z.ndim != 2:
            raise ValueError("KarplusStrongDecoder expects input (B, D)")

        # predict params
        f0_hz, decay, bright, mix = self._params_from_embedding(z)  # each (B,)

        waves = []
        # synth sample-by-sample per item (batch small in this project)
        for i in range(z.size(0)):
            w = self._ks_render_np(
                float(f0_hz[i].detach().cpu().item()),
                float(decay[i].detach().cpu().item()),
                float(bright[i].detach().cpu().item()),
                float(mix[i].detach().cpu().item()),
            )
            waves.append(torch.from_numpy(w))

        return torch.stack(waves, dim=0)  # (B, T)
