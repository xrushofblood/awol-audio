import os, argparse, yaml, numpy as np
from glob import glob

def stats(x):
    return f"shape={tuple(x.shape)}, min={x.min():.3f}, max={x.max():.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    npz_dir = cfg["paths"]["npz_out"]
    files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    assert files, f"No NPZ found in {npz_dir}"

    for fp in files[:args.limit]:
        d = np.load(fp, allow_pickle=True)
        T = d["mel"].shape[1]
        print(f"\n[FILE] {os.path.basename(fp)}")
        print(f"  mel:     {stats(d['mel'])}")
        print(f"  f0:      {stats(d['f0'])} (T={len(d['f0'])}, mel.T={T})")
        print(f"  vuv:     {stats(d['vuv'])}")
        print(f"  loud:    {stats(d['loud'])}")
        if "mel_h" in d and "mel_p" in d:
            print(f"  mel_h:   {stats(d['mel_h'])}")
            print(f"  mel_p:   {stats(d['mel_p'])}")
        print(f"  sr={int(d['sr'])}, hop={int(d['hop'])}, n_mels={int(d['n_mels'])}")

if __name__ == "__main__":
    main()
