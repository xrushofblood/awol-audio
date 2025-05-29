import numpy as np
import soundfile as sf
import os
import sys

SAMPLE_RATE = 16000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_HARMONICS = 10  # Limit harmonics for clarity

def synthesize_harmonic_sine(f0_curve, amp_curve, harmonic_distribution):
    t = np.linspace(0, DURATION, N_SAMPLES)

    # Resample curves
    f0 = np.interp(t, np.linspace(0, DURATION, len(f0_curve)), f0_curve * 600 + 100)
    amp = np.interp(t, np.linspace(0, DURATION, len(amp_curve)), amp_curve)

    # Use only first N harmonics
    harmonics = harmonic_distribution[:N_HARMONICS]
    harmonics = harmonics / (np.sum(harmonics) + 1e-6)  # Normalize

    signal = np.zeros_like(t)
    for i, h in enumerate(harmonics, 1):  # 1st harmonic is fundamental
        signal += h * np.sin(2 * np.pi * f0 * i * t)

    signal *= amp
    signal /= np.max(np.abs(signal) + 1e-6)  # Normalize final amplitude
    return signal.astype(np.float32)

def run_mock_synth(param_path, output_path=None):
    params = np.load(param_path)

    f0 = params[:32]
    amp = params[32:64]
    harmonics = params[64:]

    signal = synthesize_harmonic_sine(f0, amp, harmonics)

    if output_path is None:
        base = os.path.basename(param_path).replace("params_", "").replace(".npy", ".wav")
        output_path = os.path.join("output", "audio", base)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, signal, SAMPLE_RATE)
    print(f"Saved enhanced audio to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.audio.mock_synth <params.npy> [output.wav]")
        sys.exit(1)

    param_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    run_mock_synth(param_path, output_path)
