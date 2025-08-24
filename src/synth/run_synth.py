# src/synth/run_synth.py
import argparse
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import yaml


# -----------------------------
# Prompt → rule-based parameters
# -----------------------------
def rule_params_from_prompt(prompt: str) -> Dict[str, float]:
    """
    Very simple heuristics to map text prompt to synthesis parameters.
    Returns a dict with:
      f0_hz, decay_s, brightness (0..1), material_mix (metallic↔wood), vibrato_depth, vibrato_rate
    """
    p = prompt.lower()

    # --- pitch bucket
    if any(k in p for k in ["high-pitch", "high pitch", "very high", "soprano", "bright lead"]):
        f0_hz = 660.0
    elif any(k in p for k in ["low-pitch", "low pitch", "very low", "bass"]):
        f0_hz = 165.0
    else:
        # defaults for “pluck” mid register
        f0_hz = 330.0

    # explicit numeric hint like "f0=440" or "440hz"
    m = re.search(r"(\d+)\s*hz", p)
    if m:
        try:
            f0_hz = float(m.group(1))
        except Exception:
            pass

    # --- decay
    if any(k in p for k in ["very short", "very-short", "staccatissimo"]):
        decay_s = 0.12
    elif any(k in p for k in ["short decay", "short sustain", "staccato", "short"]):
        decay_s = 0.20
    elif any(k in p for k in ["medium decay", "medium sustain", "medium"]):
        decay_s = 0.40
    elif any(k in p for k in ["long sustain", "long decay", "ringing", "ring"]):
        decay_s = 0.80
    else:
        decay_s = 0.30

    # --- brightness / material
    brightness = 0.5
    if any(k in p for k in ["very bright", "sparkling", "sharp", "crisp"]):
        brightness = 0.9
    elif any(k in p for k in ["bright", "metallic", "glassy"]):
        brightness = 0.75
    elif any(k in p for k in ["dark", "mellow", "dull", "soft"]):
        brightness = 0.30
    elif any(k in p for k in ["warm", "wooden"]):
        brightness = 0.45

    # material mix: 1.0 metallic ↔ 0.0 wooden
    if "metallic" in p:
        material_mix = 0.85
    elif "wooden" in p or "warm" in p:
        material_mix = 0.15
    else:
        material_mix = 0.5

    # vibrato
    if any(k in p for k in ["tremolo", "vibrato"]):
        vibrato_depth = 0.01  # semitone-ish
        vibrato_rate = 6.0
    else:
        vibrato_depth = 0.0
        vibrato_rate = 5.0

    return dict(
        f0_hz=float(f0_hz),
        decay_s=float(decay_s),
        brightness=float(brightness),
        material_mix=float(material_mix),
        vibrato_depth=float(vibrato_depth),
        vibrato_rate=float(vibrato_rate),
    )


# -----------------------------
# Karplus–Strong pluck (numpy)
# -----------------------------
def _t60_to_damping(t60: float, sr: int, delay: int) -> float:
    """
    Convert a desired T60 (seconds) to a per-sample damping factor for a KS loop of given delay.
    """
    if t60 <= 1e-6:
        return 0.98
    # Rough mapping: after T60 seconds, amplitude ~ 10^(-3)
    # One loop is 'delay' samples; per-loop gain gL; total loops ≈ sr * T60 / delay.
    loops = max(1.0, (sr * t60) / max(1, delay))
    g_loop = 10.0 ** (-3.0 / loops)
    # Spread per-sample (approximate) – allow slight extra loss in the 2-tap averager
    g_sample = g_loop ** (1.0 / max(1, delay))
    return float(np.clip(g_sample, 0.90, 0.9999))


def ks_synthesize(
    sr: int,
    seconds: float,
    f0_hz: float,
    decay_s: float,
    brightness: float,
    material_mix: float,
    vibrato_depth: float = 0.0,
    vibrato_rate: float = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Minimal Karplus–Strong plucked-string synthesis with a simple loop filter:
      - delay line with 2-tap (averager) + damping
      - one-pole low-pass whose cutoff depends on 'brightness' and 'material_mix'
      - optional simple vibrato (sinusoidal delay modulation)
    Returns mono float32 audio in [-1, 1].
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sr * seconds)
    n_samples = max(n_samples, 1)

    # Delay (ensure >= 2)
    delay = max(2, int(round(sr / max(30.0, min(4000.0, f0_hz)))))
    buf = np.zeros(delay, dtype=np.float32)

    # Excitation noise burst length ~ small fraction of a period
    exc_len = min(delay, max(8, int(0.5 * delay)))
    # Mix of white & band-limited noise depending on "material"
    white = rng.uniform(-1.0, 1.0, size=exc_len).astype(np.float32)
    # simple 1st-order LP for band-limit (wooden/pluck softer)
    lp_a = 1.0 - 0.15 * (1.0 - brightness)  # higher -> brighter
    band = np.zeros_like(white)
    z = 0.0
    for i, w in enumerate(white):
        z = lp_a * z + (1.0 - lp_a) * w
        band[i] = z
    exc = (material_mix * white + (1.0 - material_mix) * band).astype(np.float32)

    # load the buffer with the excitation (with tiny DC removal)
    exc -= np.mean(exc)
    buf[:exc_len] = exc

    # per-sample damping from desired decay time
    g = _t60_to_damping(decay_s, sr, delay)

    # loop low-pass coefficient from brightness (0 = very dark, 1 = very bright)
    # map to alpha in [0.15, 0.95]
    alpha = 0.15 + 0.80 * float(np.clip(brightness, 0.0, 1.0))

    out = np.zeros(n_samples, dtype=np.float32)
    # Internal states
    y_prev = 0.0
    lp_state = 0.0

    # vibrato in samples (peak deviation ~ depth in semitones)
    if vibrato_depth > 0.0:
        semitone_ratio = 2 ** (vibrato_depth / 12.0)  # e.g., ~1.0008 for 0.01
        max_dev = int(round(delay * (semitone_ratio - 1.0)))
        max_dev = max(0, min(max_dev, delay // 8))
        vib_lfo = 2.0 * math.pi * vibrato_rate / sr
    else:
        max_dev = 0
        vib_lfo = 0.0

    # main loop
    widx = 0  # write index
    for n in range(n_samples):
        # read position (optionally modulated)
        if max_dev > 0:
            dev = int(round(max_dev * math.sin(vib_lfo * n)))
        else:
            dev = 0
        ridx = (widx - (delay + dev)) % delay
        ridx_prev = (ridx - 1) % delay

        # 2-tap averager + damping
        y = 0.5 * (buf[ridx] + buf[ridx_prev])
        y *= g

        # simple loop low-pass to control brightness (tone)
        lp_state = alpha * y + (1.0 - alpha) * lp_state
        y = lp_state

        out[n] = y
        # update buffer (string) with new sample + a tiny loss via averager
        buf[widx] = 0.5 * (y + y_prev)
        y_prev = y
        widx = (widx + 1) % delay

    # normalize mildly to prevent huge peaks; keep pluck feel
    peak = float(np.max(np.abs(out)) + 1e-9)
    out = 0.90 * out / peak
    return out.astype(np.float32)


# -----------------------------
# I/O helpers
# -----------------------------
def save_wave(path: Path, y: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="PCM_16")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Rule-based Karplus–Strong pluck from text prompt.")
    ap.add_argument("--config", default="configs/synth.yaml", help="YAML with synth paths and settings")
    ap.add_argument("--prompt", required=True, help='e.g., "bright metallic pluck with short decay"')
    ap.add_argument("--seed", type=int, default=0, help="Random seed for excitation")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, "r"))
    sr = int(cfg.get("synth", {}).get("sample_rate", 16000))
    seconds = float(cfg.get("synth", {}).get("seconds", 1.0))
    outdir = Path(cfg.get("paths", {}).get("output_dir", "outputs/gen_audio"))

    # Prompt → params
    params = rule_params_from_prompt(args.prompt)

    # Render
    y = ks_synthesize(
        sr=sr,
        seconds=seconds,
        f0_hz=params["f0_hz"],
        decay_s=params["decay_s"],
        brightness=params["brightness"],
        material_mix=params["material_mix"],
        vibrato_depth=params["vibrato_depth"],
        vibrato_rate=params["vibrato_rate"],
        seed=args.seed,
    )

    # Save + dbg
    fname = outdir / f"{args.prompt.replace(' ', '_')}.wav"
    save_wave(fname, y, sr)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12)
    print(f"[Day7 KS] Saved audio to: {fname}")
    print(
        "[DBG] audio shape:", y.shape,
        "min:", float(y.min()),
        "max:", float(y.max()),
        "rms:", rms,
        "| params:", params
    )


if __name__ == "__main__":
    main()
