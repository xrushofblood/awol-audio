# src/synth/predict_params.py
import argparse, yaml, numpy as np, torch
from pathlib import Path
from .param_regressor import ParamRegressor, ParamBounds, denorm_to_dict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/params_real.yaml")
    ap.add_argument("--ckpt",   default="checkpoints/paramreg/paramreg_best.pt")
    ap.add_argument("--emb",    required=True, help="path to a single audio embedding (.npy), 512D")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,"r"))
    bounds = ParamBounds.from_cfg(cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = ParamRegressor(cfg["model"]["in_dim"], cfg["model"]["hidden"], cfg["model"]["out_dim"], cfg["model"]["dropout"])
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state); model.eval()

    x = np.load(args.emb).astype("float32")
    x = x / (np.linalg.norm(x) + 1e-8)
    with torch.no_grad():
        y01 = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0)
    params = denorm_to_dict(y01, bounds)
    print(params)

if __name__ == "__main__":
    main()
