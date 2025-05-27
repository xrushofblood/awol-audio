import os
import sys
import subprocess

embedding_dir = "data/embeddings"
model_path = "models/mapper.pt"
output_dir = "output/params"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(embedding_dir):
    if file.endswith(".npy") and file.startswith("embedding_"):
        embedding_path = os.path.join(embedding_dir, file)
        filename = file.replace("embedding_", "params_")
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            print(f"Skipping (already exists): {output_path}")
            continue

        print(f"Generating: {output_path}")
        subprocess.run([
            sys.executable,
            "-m",
            "src.mapping.inference_mapper",
            embedding_path,
            model_path,
            output_path
        ])

