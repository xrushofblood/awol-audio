# src/mapper/train_mapper.py
import os, json, argparse, yaml, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .mapper_mlp import MLPMapper

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

class PairDataset(Dataset):
    """Pairs text_emb (.text.npy) with audio_emb (.audio.npy) by filename stem."""
    def __init__(self, text_dir, audio_dir, names):
        self.text_dir = Path(text_dir); self.audio_dir = Path(audio_dir)
        self.names = names
        # Infer dim sanity checks
        td = json.load(open(self.text_dir / "meta.json"))["dim"]
        ad = json.load(open(self.audio_dir / "meta.json"))["dim"]
        assert td == ad, f"Dim mismatch text={td} audio={ad}"
        self.dim = td

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        t = np.load(self.text_dir / f"{name}.text.npy").astype(np.float32)
        a = np.load(self.audio_dir / f"{name}.audio.npy").astype(np.float32)
        # (Optional) L2 normalize both
        t = t / (np.linalg.norm(t) + 1e-9)
        a = a / (np.linalg.norm(a) + 1e-9)
        return torch.from_numpy(t), torch.from_numpy(a), name

def device_from_cfg(cfg):
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"

def make_pairs_lists(text_dir, audio_dir, val_ratio, seed):
    text_files = {p.stem.replace(".text",""): p for p in Path(text_dir).glob("*.text.npy")}
    audio_files = {p.stem.replace(".audio",""): p for p in Path(audio_dir).glob("*.audio.npy")}
    common = sorted(list(set(text_files.keys()) & set(audio_files.keys())))
    assert len(common) > 0, "No paired embeddings found."
    train_names, val_names = train_test_split(common, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_names, val_names

def make_loss(name):
    if name == "cosine":
        # minimize (1 - cosine)
        return lambda pred, tgt: (1.0 - nn.functional.cosine_similarity(pred, tgt, dim=-1)).mean()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)
    set_seed(cfg["train"]["seed"])

    text_dir = cfg["paths"]["text_emb_dir"]
    audio_dir = cfg["paths"]["audio_emb_dir"]
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_names, val_names = make_pairs_lists(text_dir, audio_dir, cfg["train"]["val_ratio"], cfg["train"]["seed"])
    dtrain = PairDataset(text_dir, audio_dir, train_names)
    dval   = PairDataset(text_dir, audio_dir, val_names)

    train_loader = DataLoader(dtrain, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False)
    val_loader   = DataLoader(dval,   batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False)

    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"], hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = make_loss(cfg["train"]["loss"])

    best_val = 1e9
    patience = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train(); tr_loss = 0.0; ntr = 0
        for t, a, _ in train_loader:
            t = t.to(device); a = a.to(device)
            p = model(t)
            loss = criterion(p, a)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * t.size(0); ntr += t.size(0)

        model.eval(); va_loss = 0.0; nva = 0
        with torch.no_grad():
            for t, a, _ in val_loader:
                t = t.to(device); a = a.to(device)
                p = model(t)
                loss = criterion(p, a)
                va_loss += loss.item() * t.size(0); nva += t.size(0)

        tr = tr_loss / max(1, ntr)
        va = va_loss / max(1, nva)
        print(f"[EPOCH {epoch:03d}] train_loss={tr:.4f} | val_loss={va:.4f}")

        if va < best_val:
            best_val = va; patience = 0
            ckpt_path = ckpt_dir / "mapper_best.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
            print(f"[CKPT] Saved best to {ckpt_path}")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("[EARLY STOP]")
                break

if __name__ == "__main__":
    main()
