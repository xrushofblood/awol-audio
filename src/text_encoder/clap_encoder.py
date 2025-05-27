from transformers import ClapProcessor, ClapModel
import torch
import numpy as np
import os

def get_clap_embedding(prompt: str, model, processor) -> np.ndarray:
    inputs = processor(text=[prompt], return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_text_features(**inputs)
    return embedding[0].cpu().numpy()

def save_embedding(prompt: str, embedding: np.ndarray, save_dir: str = "data/embeddings") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = prompt.lower().replace(" ", "_").replace("/", "-")
    path = os.path.join(save_dir, f"embedding_{filename}.npy")
    np.save(path, embedding)
    return path

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python clap_encoder.py \"your prompt here\"")
        exit()

    prompt = sys.argv[1]
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True)
    embedding = get_clap_embedding(prompt, model, processor)
    save_embedding(prompt, embedding)
