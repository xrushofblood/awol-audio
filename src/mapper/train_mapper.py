# src/mapper/train_mapper.py
import os, json, argparse, yaml, random, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .mapper_mlp import MLPMapper

# usiamo il param-regressor e l'estrazione target già esistenti
from src.synth.param_regressor import ParamRegressor
from src.synth.targets_from_npz import targets_from_npz


EPS = 1e-9


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def device_from_cfg(cfg):
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if (d == "cuda" and torch.cuda.is_available()) else "cpu"


def l2norm_np(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + EPS
    return x / n


def l2norm_torch(x: torch.Tensor):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def to_unit(x, spec):
    """
    Mappa valore reale -> [0,1] secondo spec {min,max,scale?}.
    scale: 'linear' (default) o 'log_hz' (per pitch in Hz).
    """
    lo, hi = float(spec["min"]), float(spec["max"])
    scale = spec.get("scale", "linear")
    x = np.asarray(x, dtype=np.float32)

    if scale == "log_hz":
        x_clip = np.clip(x, lo, hi)
        x_log  = np.log2(x_clip + 1e-12)
        lo_log = math.log2(lo); hi_log = math.log2(hi)
        return (x_log - lo_log) / (hi_log - lo_log + 1e-9)
    else:
        return (x - lo) / (hi - lo + 1e-9)


def make_pairs_lists(text_dir, audio_dir, val_ratio, seed):
    text_files = {p.stem.replace(".text",""): p for p in Path(text_dir).glob("*.text.npy")}
    audio_files = {p.stem.replace(".audio",""): p for p in Path(audio_dir).glob("*.audio.npy")}
    common = sorted(list(set(text_files.keys()) & set(audio_files.keys())))
    assert len(common) > 0, "No paired embeddings found."
    train_names, val_names = train_test_split(common, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_names, val_names


class PairDataset(Dataset):
    """
    Coppie (text_emb, audio_emb) + target parametrici normalizzati [0,1] per loss ausiliaria.
    """
    def __init__(self, text_dir, audio_dir, npz_dir, names, audio_cfg, param_specs):
        self.text_dir = Path(text_dir)
        self.audio_dir = Path(audio_dir)
        self.npz_dir   = Path(npz_dir)
        self.names     = names
        self.audio_cfg = audio_cfg
        self.param_specs = param_specs

        # sanity: le dimensioni di text/audio nel meta devono combaciare
        td = json.load(open(self.text_dir / "meta.json"))["dim"]
        ad = json.load(open(self.audio_dir / "meta.json"))["dim"]
        assert td == ad, f"Dim mismatch text={td} audio={ad}"
        self.dim = td

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        t = np.load(self.text_dir / f"{name}.text.npy").astype(np.float32)   # (D,)
        a = np.load(self.audio_dir / f"{name}.audio.npy").astype(np.float32) # (D,)

        # L2 normalize embeddings (coerente con tutto il resto della pipeline)
        t = l2norm_np(t)
        a = l2norm_np(a)

        # targets parametrici reali -> [0,1] (stesso schema del paramreg)
        d = np.load(self.npz_dir / f"{name}.npz")
        y_real = targets_from_npz(
            d,
            sr=self.audio_cfg["sample_rate"],
            hop=self.audio_cfg["hop_size"],
            fmin_hz=self.audio_cfg.get("fmin_hz", 30.0),
            fmax_hz=self.audio_cfg.get("fmax_hz", 12000.0),
        )  # shape (P,)
        y01 = np.zeros_like(y_real, dtype=np.float32)
        for j, spec in enumerate(self.param_specs):
            y01[j] = np.clip(to_unit(y_real[j], spec), 0.0, 1.0)

        return torch.from_numpy(t), torch.from_numpy(a), torch.from_numpy(y01), name


def make_loss(name):
    if name == "cosine":
        # minimizziamo (1 - cosine)
        return lambda pred, tgt: (1.0 - nn.functional.cosine_similarity(pred, tgt, dim=-1)).mean()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {name}")


def make_param_loss(name, beta=0.05):
    if name == "huber":
        return nn.SmoothL1Loss(beta=beta)
    elif name == "l1":
        return nn.L1Loss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown param loss: {name}")


def load_paramreg_frozen(ckpt_path, model_cfg):
    """
    Carica il ParamRegressor congelato (output in [0,1]) per la loss ausiliaria.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ParamRegressor(
        in_dim=model_cfg["in_dim"],
        hidden=model_cfg["hidden"],
        out_dim=model_cfg["out_dim"],
        dropout=model_cfg["dropout"],
    )
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg)
    set_seed(cfg["train"]["seed"])

    # --- paths & cfgs
    text_dir = cfg["paths"]["text_emb_dir"]
    audio_dir = cfg["paths"]["audio_emb_dir"]
    npz_dir   = cfg["paths"]["npz_dir"]               # <== nuovo
    ckpt_dir  = Path(cfg["paths"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # config dei parametri/synth (per normalizzazione [0,1])
    # Se preferisci, puoi mettere direttamente "synth: { params: [...] }" dentro questo yaml,
    # ma qui carichiamo dal params_real.yaml per coerenza.
    params_cfg_path = cfg["paths"].get("params_config", None)
    if params_cfg_path is None:
        raise ValueError("Missing paths.params_config in mapper config (serve per specs dei parametri).")
    params_cfg = yaml.safe_load(open(params_cfg_path, "r"))
    param_specs = params_cfg["synth"]["params"]
    audio_cfg   = params_cfg["audio"]

    # split train/val
    train_names, val_names = make_pairs_lists(text_dir, audio_dir, cfg["train"]["val_ratio"], cfg["train"]["seed"])
    dtrain = PairDataset(text_dir, audio_dir, npz_dir, train_names, audio_cfg, param_specs)
    dval   = PairDataset(text_dir, audio_dir, npz_dir, val_names,   audio_cfg, param_specs)

    train_loader = DataLoader(dtrain, batch_size=cfg["train"]["batch_size"], shuffle=True,  drop_last=False)
    val_loader   = DataLoader(dval,   batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False)

    # mapper
    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"],
        hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"],
        dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    # perdita in spazio embedding
    emb_loss_fn = make_loss(cfg["train"]["loss"])

    # carichiamo il paramreg congelato per la loss ausiliaria
    paramreg_ckpt = cfg["paths"].get("paramreg_ckpt", None)
    if paramreg_ckpt is None or not Path(paramreg_ckpt).exists():
        raise ValueError("paths.paramreg_ckpt mancante o inesistente: serve per la param-aux loss.")
    paramreg = load_paramreg_frozen(paramreg_ckpt, params_cfg["model"]).to(device)

    # perdita sui parametri (in [0,1])
    param_loss_fn = make_param_loss(cfg["train"].get("param_loss", "huber"),
                                    beta=float(cfg["train"].get("param_beta", 0.05)))

    # pesi
    lam_emb   = float(cfg["train"].get("lambda_embed", 0.5))
    lam_param = float(cfg["train"].get("lambda_param", 0.5))

    best_val = 1e9
    patience = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        # -------------------- TRAIN --------------------
        model.train()
        tr_e, tr_p, tr_tot, ntr = 0.0, 0.0, 0.0, 0
        for t, a, y01, _ in train_loader:
            t = t.to(device)                # (B, D)
            a = a.to(device)                # (B, D)
            y01 = y01.to(device)            # (B, P)

            p = model(t)                    # (B, D)
            # loss embedding
            loss_emb = emb_loss_fn(p, a)

            # loss parametrica: normalizza p -> passa nel paramreg (congelato)
            p_n = l2norm_torch(p)
            y_pred01 = paramreg(p_n)        # (B, P) in [0,1]
            loss_param = param_loss_fn(y_pred01, y01)

            loss = lam_emb * loss_emb + lam_param * loss_param

            opt.zero_grad(); loss.backward(); opt.step()

            bs = t.size(0)
            tr_e   += loss_emb.item()  * bs
            tr_p   += loss_param.item()* bs
            tr_tot += loss.item()      * bs
            ntr    += bs

        # -------------------- VAL ----------------------
        model.eval()
        va_e, va_p, va_tot, nva = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for t, a, y01, _ in val_loader:
                t = t.to(device); a = a.to(device); y01 = y01.to(device)
                p = model(t)
                loss_emb = emb_loss_fn(p, a)
                p_n = l2norm_torch(p)
                y_pred01 = paramreg(p_n)
                loss_param = param_loss_fn(y_pred01, y01)
                loss = lam_emb * loss_emb + lam_param * loss_param

                bs = t.size(0)
                va_e   += loss_emb.item()  * bs
                va_p   += loss_param.item()* bs
                va_tot += loss.item()      * bs
                nva    += bs

        trE, trP, trT = tr_e/max(1,ntr), tr_p/max(1,ntr), tr_tot/max(1,ntr)
        vaE, vaP, vaT = va_e/max(1,nva), va_p/max(1,nva), va_tot/max(1,nva)
        print(f"[EPOCH {epoch:03d}] "
              f"train: emb={trE:.4f} param={trP:.4f} total={trT:.4f} | "
              f"val: emb={vaE:.4f} param={vaP:.4f} total={vaT:.4f}")

        # early stop sulla loss totale di validazione
        if vaT < best_val - 1e-6:
            best_val = vaT; patience = 0
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
