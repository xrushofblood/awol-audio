import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.mapping.mapper import MapperMLP

# Step 1: Load CLAP embeddings
def load_embeddings(embedding_dir):
    X = []
    for file in os.listdir(embedding_dir):
        if file.endswith(".npy") and file.startswith("embedding_"):
            emb = np.load(os.path.join(embedding_dir, file))
            X.append(emb)
    return np.stack(X)

# Step 2: Generate synthetic targets
def generate_targets(n_samples, dim=128):
    t = np.linspace(0, 1, dim)
    Y = []
    for i in range(n_samples):
        f0 = 0.5 * np.sin(2 * np.pi * t * (i+1)) + 0.5
        amplitude = np.exp(-5 * t)
        center = int(dim * 0.33)
        harmonic = np.exp(-0.5 * ((np.arange(dim) - center) / 15) ** 2)
        harmonic = harmonic / harmonic.sum()
        sample = np.concatenate([f0[:32], amplitude[:32], harmonic[:64]])
        Y.append(sample)
    return np.stack(Y)

# Step 3: Train
def train(model, X_tensor, Y_tensor, n_epochs=200, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return model, losses

if __name__ == "__main__":
    # Config
    embedding_dir = "data/embeddings"
    model_save_path = "models/mapper.pt"
    os.makedirs("models", exist_ok=True)

    print("Loading embeddings...")
    X = load_embeddings(embedding_dir)
    Y = generate_targets(X.shape[0], dim=128)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    print("Initializing model...")
    model = MapperMLP(input_dim=512, hidden_dim=256, output_dim=128)

    print("Training model...")
    model, losses = train(model, X_tensor, Y_tensor, n_epochs=200, lr=0.001)

    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)
    print("Done!")
