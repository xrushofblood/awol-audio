from transformers import ClapProcessor, ClapModel
from src.text_encoder.clap_encoder import get_clap_embedding, save_embedding
import os

def load_prompts(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    prompt_file = os.path.join(project_root, "data", "embeddings", "prompts.txt")
    save_dir = os.path.join(project_root, "data", "embeddings")


    print("Loading CLAP model...")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused", use_safetensors=True)

    prompts = load_prompts(prompt_file)
    print(f"Found {len(prompts)} prompts")

    for prompt in prompts:
        print(f"Generating: \"{prompt}\"")
        embedding = get_clap_embedding(prompt, model, processor)
        path = save_embedding(prompt, embedding, save_dir)
        print(f"Saved to: {path}")

    print("\nAll embeddings generated.")
