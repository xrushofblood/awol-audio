# scripts/build_faiss_index.py
# Build a FAISS inner-product index over your *audio* embeddings.
# IMPORTANT:
#  - We L2-normalize vectors here, so inner product = cosine similarity.
#  - Filenames saved to names.npy must match the stems your mapper eval uses.

import glob
import numpy as np
import faiss
from pathlib import Path

# Adjust if your layout is different
AUDIO_DIR = Path("data/processed/embeddings/audio")
OUT_DIR   = Path("data/processed/index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Collect all audio embedding files (*.audio.npy)
files = sorted(glob.glob(str(AUDIO_DIR / "*.audio.npy")))
assert files, f"No *.audio.npy found in {AUDIO_DIR}"

# Load embeddings and make matching names (stems WITHOUT ".audio")
X = [np.load(f).astype("float32") for f in files]
names = [Path(f).stem.replace(".audio", "") for f in files]

X = np.stack(X, axis=0)  # (N, D)

# L2-normalize so that inner product equals cosine similarity
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

# Build a simple flat IP index (cosine because we normalized X)
index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

# Persist index and names
faiss.write_index(index, str(OUT_DIR / "faiss_ip.index"))
np.save(OUT_DIR / "names.npy", np.array(names, dtype=object))

print(f"[OK] Built FAISS IP index with {len(names)} vectors -> {OUT_DIR}")
