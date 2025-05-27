import torch
import numpy as np
import os
import sys

from src.mapping.mapper import MapperMLP

def load_embedding(file_path):
    """Load a .npy embedding file from disk."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)

def run_inference(embedding_path, model_path, output_path=None):
    # Load embedding
    embedding = load_embedding(embedding_path)
    x_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # shape: [1, 512]

    # Load model
    model = MapperMLP(input_dim=512, hidden_dim=256, output_dim=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Run inference
    with torch.no_grad():
        output = model(x_tensor).squeeze(0).numpy()

    # Save or print
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, output)
        print(f"Output saved to {output_path}")
    else:
        print("Generated parameter vector:")
        print(output[:10], "...")  # Show first 10 values

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/mapping/inference_mapper.py <embedding.npy> <mapper.pt> [output.npy]")
        sys.exit(1)

    embedding_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    run_inference(embedding_path, model_path, output_path)
