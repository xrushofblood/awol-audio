# src/mapper/mapper_smoke.py
import yaml, torch
from pathlib import Path
from .train_mapper import PairDataset, make_pairs_lists, device_from_cfg, set_seed
from .mapper_mlp import MLPMapper
from torch.utils.data import DataLoader
import torch.nn as nn

def main():
    cfg = yaml.safe_load(open("configs/mapper.yaml"))
    device = device_from_cfg(cfg); set_seed(0)
    text_dir = cfg["paths"]["text_emb_dir"]; audio_dir = cfg["paths"]["audio_emb_dir"]
    train_names, _ = make_pairs_lists(text_dir, audio_dir, 0.2, 0)
    dtrain = PairDataset(text_dir, audio_dir, train_names[:16])
    loader = DataLoader(dtrain, batch_size=8, shuffle=True)

    model = MLPMapper().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = lambda p,a: (1 - nn.functional.cosine_similarity(p,a,dim=-1)).mean()

    model.train()
    for epoch in range(2):
        total = 0.0; n = 0
        for t,a,_ in loader:
            t, a = t.to(device), a.to(device)
            p = model(t)
            loss = loss_fn(p, a)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * t.size(0); n += t.size(0)
        print(f"[SMOKE] epoch={epoch} loss={total/max(1,n):.4f}")

if __name__ == "__main__":
    main()
