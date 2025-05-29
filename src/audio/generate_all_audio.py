import os
import sys
import subprocess

param_dir = "output/params"
audio_dir = "output/audio"

os.makedirs(audio_dir, exist_ok=True)

for file in os.listdir(param_dir):
    if file.endswith(".npy") and file.startswith("params_"):
        param_path = os.path.join(param_dir, file)
        base_name = file.replace("params_", "").replace(".npy", ".wav")
        output_path = os.path.join(audio_dir, base_name)

        if os.path.exists(output_path):
            print(f"Skipping existing audio: {output_path}")
            continue

        print(f"🎧 Synthesizing: {output_path}")
        subprocess.run([
            sys.executable,
            "-m",
            "src.audio.mock_synth",
            param_path,
            output_path
        ])
