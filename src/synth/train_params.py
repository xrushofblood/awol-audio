# src/synth/train_params.py
import argparse, yaml, glob, math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .param_regressor import ParamRegressor
from .targets_from_npz import targets_from_npz

EPS = 1e-8

def l2norm_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + EPS
    return x / n

def load_perfile_embeddings(audio_emb_dir):
    files = sorted(glob.glob(str(Path(audio_emb_dir) / "*.audio.npy")))
    assert files, f"No *.audio.npy in {audio_emb_dir}"
    X = [np.load(f).astype("float32") for f in files]
    names = [Path(f).stem.replace(".audio", "") for f in files]
    return np.stack(X, axis=0), names  # (N,D), list[str]

def to_unit(x, spec):
    """Map real-valued target -> [0,1] using spec {min,max,scale?}."""
    lo, hi = float(spec["min"]), float(spec["max"])
    scale = spec.get("scale", "linear")
    x = np.asarray(x, dtype=np.float32)
    if scale == "log_hz":
        x = np.log2(np.clip(x, lo, hi))
        lo_log, hi_log = math.log2(lo), math.log2(hi)
        return (x - lo_log) / (hi_log - lo_log + 1e-9)
    else:
        return (x - lo) / (hi - lo + 1e-9)

def get_spec(cfg, name):
    for s in cfg["synth"]["params"]:
        if s["name"] == name:
            return s
    raise KeyError(f"Param spec '{name}' not found")

class ParamDataset(Dataset):
    """Pairs CLAP audio embeddings with parameter targets derived from .npz."""
    def __init__(self, cfg):
        self.cfg = cfg
        audio_emb_dir = Path(cfg["paths"]["audio_emb_dir"])
        npz_dir       = Path(cfg["paths"]["npz_dir"])

        # inputs
        X, names = load_perfile_embeddings(audio_emb_dir)

        # get decay spec to clamp labels coherently with YAML
        decay_spec = get_spec(cfg, "decay_t60")
        dmin, dmax = float(decay_spec["min"]), float(decay_spec["max"])

        # targets (real units) -> normalize per spec to [0,1]
        specs = cfg["synth"]["params"]
        Y_list = []
        for name in names:
            npz_path = npz_dir / f"{name}.npz"
            assert npz_path.exists(), f"Missing npz for {name}: {npz_path}"
            d = np.load(npz_path)
            y = targets_from_npz(
                d,
                sr=cfg["audio"]["sample_rate"],
                hop=cfg["audio"]["hop_size"],
                fmin_hz=cfg["audio"].get("fmin_hz", 30.0),
                fmax_hz=cfg["audio"].get("fmax_hz", 12000.0),
                decay_min=dmin, decay_max=dmax,   # <-- clamp here
            )
            Y_list.append(y)
        Y_real = np.stack(Y_list, axis=0).astype("float32")  # (N,P)

        Y01 = np.zeros_like(Y_real, dtype=np.float32)
        for j, spec in enumerate(specs):
            Y01[:, j] = np.clip(to_unit(Y_real[:, j], spec), 0.0, 1.0)

        # store
        self.X   = l2norm_np(X).astype("float32")
        self.Y01 = Y01.astype("float32")

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y01[i]

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    if torch.is_tensor(x):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)

def train_one(cfg):
    # --- robust casting ---
    t = cfg["train"]
    t["lr"] = float(t["lr"])
    t["weight_decay"] = float(t["weight_decay"])
    t["batch_size"] = int(t["batch_size"])
    t["epochs"] = int(t["epochs"])
    t["val_ratio"] = float(t["val_ratio"])
    t["early_stop_patience"] = int(t["early_stop_patience"])
    t["embed_noise_std"] = float(t.get("embed_noise_std", 0.0))
    loss_name = str(t.get("loss", "mse")).lower()
    loss_beta = float(t.get("loss_beta", 0.05))
    w_map = t.get("loss_weights", {})  # dict param_name -> weight

    # dataset & split
    ds = ParamDataset(cfg)
    idx = np.arange(len(ds))
    tr_idx, va_idx = train_test_split(
        idx, test_size=t["val_ratio"], random_state=t["seed"], shuffle=True
    )

    def make_loader(indices, shuffle):
        subset = torch.utils.data.Subset(ds, indices.tolist())
        return DataLoader(subset, batch_size=t["batch_size"], shuffle=shuffle, drop_last=False)

    tr_loader = make_loader(tr_idx, True)
    va_loader = make_loader(va_idx, False)

    # model
    model = ParamRegressor(
        in_dim=cfg["model"]["in_dim"],
        hidden=cfg["model"]["hidden"],
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
    )
    device = "cuda" if (torch.cuda.is_available() and cfg.get("device","cpu")=="cuda") else "cpu"
    model.to(device)

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    # base loss
    if loss_name == "huber":
        base_loss = nn.SmoothL1Loss(beta=loss_beta, reduction="none")
    elif loss_name == "l1":
        base_loss = nn.L1Loss(reduction="none")
    else:
        base_loss = nn.MSELoss(reduction="none")

    # per-parameter weights in the same order as cfg["synth"]["params"]
    specs = cfg["synth"]["params"]
    names_order = [p["name"] for p in specs]
    weights = torch.tensor([float(w_map.get(n, 1.0)) for n in names_order], dtype=torch.float32)

    def crit_weighted(pred, target):
        per_elem = base_loss(pred, target)  # (B,P)
        return (per_elem * weights.to(per_elem.device)).mean()

    # loop + early stop
    best_va, patience = 1e9, 0
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(t["epochs"]):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = to_tensor(xb).to(device)
            yb = to_tensor(yb).to(device)
            if t["embed_noise_std"] > 0:
                xb = xb + torch.randn_like(xb) * t["embed_noise_std"]
            pred = model(xb)
            loss = crit_weighted(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = to_tensor(xb).to(device)
                yb = to_tensor(yb).to(device)
                va_loss += crit_weighted(model(xb), yb).item() * xb.size(0)
        va_loss /= len(va_loader.dataset)

        print(f"[EPOCH {epoch+1:03d}] train={tr_loss:.5f}  val={va_loss:.5f}")

        if va_loss < best_va - 1e-5:
            best_va, patience = va_loss, 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg}, ckpt_dir / "paramreg_best.pt")
        else:
            patience += 1
            if patience >= t["early_stop_patience"]:
                print(f"[EARLY STOP] best val={best_va:.5f}")
                break

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params_real.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    train_one(cfg)

if __name__ == "__main__":
    main()
