# src/mapper/evaluate_mapper.py
import argparse, yaml, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import faiss

from .mapper_mlp import MLPMapper
from .train_mapper import PairDataset, make_pairs_lists, device_from_cfg, set_seed

def load_faiss(index_dir):
    index = faiss.read_index(str(Path(index_dir) / "faiss_ip.index"))
    names = np.load(Path(index_dir) / "names.npy", allow_pickle=True).tolist()
    return index, names

def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    return x / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mapper.yaml")
    ap.add_argument("--ckpt", default="checkpoints/mapper/mapper_best.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = device_from_cfg(cfg); set_seed(cfg["train"]["seed"])

    text_dir = cfg["paths"]["text_emb_dir"]
    audio_dir = cfg["paths"]["audio_emb_dir"]
    index_dir = cfg["paths"]["index_dir"]
    topk = cfg["eval"]["topk"]

    # dataset (val only)
    _, val_names = make_pairs_lists(text_dir, audio_dir, cfg["train"]["val_ratio"], cfg["train"]["seed"])
    dval = PairDataset(text_dir, audio_dir, val_names)
    loader = DataLoader(dval, batch_size=32, shuffle=False)

    # model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = MLPMapper(
        in_dim=cfg["model"]["in_dim"], hidden=tuple(cfg["model"]["hidden"]),
        out_dim=cfg["model"]["out_dim"], dropout=cfg["model"]["dropout"],
        norm_out=cfg["model"]["norm_out"]
    )
    model.load_state_dict(ckpt["model"]); model.to(device); model.eval()

    # FAISS
    index, names = load_faiss(index_dir)

    # metrics
    cos_list = []
    r_at_1 = r_at_5 = 0
    total = 0

    with torch.no_grad():
        for t, a, name in loader:
            t = t.to(device); a = a.to(device)
            p = model(t)
            # cosine
            cos = torch.nn.functional.cosine_similarity(p, a, dim=-1).cpu().numpy()
            cos_list.extend(cos.tolist())

            # retrieval
            p_np = p.cpu().numpy()
            p_np = l2norm(p_np).astype(np.float32)
            D, I = index.search(p_np, topk)
            for qidx in range(I.shape[0]):
                total += 1
                pred_names = [names[i] for i in I[qidx]]
                gold = name[qidx]
                if pred_names[0] == gold: r_at_1 += 1
                if gold in pred_names:    r_at_5 += 1

    cos_mean = float(np.mean(cos_list)) if cos_list else 0.0
    print(f"[EVAL] mean cosine similarity (val): {cos_mean:.4f}")
    print(f"[EVAL] R@1: {r_at_1/total:.3f} | R@5: {r_at_5/total:.3f}  (N={total})")

if __name__ == "__main__":
    main()
