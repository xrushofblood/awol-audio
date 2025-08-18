import os, yaml, numpy as np
from glob import glob

def quick_checks(cfg):
    npz_dir = cfg["paths"]["npz_out"]
    files = glob(os.path.join(npz_dir, "*.npz"))
    assert len(files) > 0, "No NPZ produced."
    for p in files[:5]:
        d = np.load(p, allow_pickle=True)
        T = d["mel"].shape[1]
        assert d["f0"].shape[0] == T, "f0 length mismatch"
        assert d["loud"].shape[0] == T, "loud length mismatch"
        assert d["mel_h"].shape[1] == T and d["mel_p"].shape[1] == T, "HPSS mel mismatch"
        # f0 in-range (voiced)
        f0 = d["f0"]; vuv = d["vuv"]
        if vuv.sum() > 0:
            assert (f0[vuv>0] > 0).all(), "Voiced frames must have f0>0"
    print(f"OK: {len(files)} NPZ files validated.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    quick_checks(cfg)
