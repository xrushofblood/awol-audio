# scripts/check_index_coverage.py
# Verify that the *validation* items used by mapper/eval exist in the FAISS index.
# It uses the SAME val split rule (val_ratio + seed) as your mapper config.

import glob
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

CFG_PATH = "configs/mapper_real.yaml"

cfg = yaml.safe_load(open(CFG_PATH, "r"))

audio_dir = Path(cfg["paths"]["audio_emb_dir"])
index_dir = Path(cfg["paths"]["index_dir"])

# Build the list of AVAILABLE stems (consistent with how we save/load)
stems = sorted([
    Path(f).stem.replace(".audio", "")
    for f in glob.glob(str(audio_dir / "*.audio.npy"))
])
assert stems, f"No *.audio.npy found in {audio_dir}"

# Reproduce the same train/val split used in training
_, val_names = train_test_split(
    stems,
    test_size=cfg["train"]["val_ratio"],
    random_state=cfg["train"]["seed"],
    shuffle=True
)

# Load index names
names = set(np.load(index_dir / "names.npy", allow_pickle=True).tolist())

# Who is missing from the index?
missing = [n for n in val_names if n not in names]

print(f"VAL size = {len(val_names)} | missing in index = {len(missing)}")
if missing:
    print("Examples missing (first 10):", missing[:10])
else:
    print("[OK] All validation items are present in the index.")
