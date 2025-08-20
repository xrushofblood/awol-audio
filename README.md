# AWOL-Audio (Text-to-Audio Prototype)

This repository implements an **AWOL-inspired pipeline** for text-to-audio generation, starting with synthetic data and progressively extending to real datasets.

---

## Current Status

### Day 1
- Synthetic dataset generation:
  - Created `make_synthetic_plucks.py` to generate short pluck sounds.
  - Associated prompts stored in `data/meta/prompts.csv`.
- Configuration:
  - `configs/base.yaml` defines sample rate, hop size, frame size, and output paths.
- Audio analysis modules (in `src/analysis/`):
  - `audio_io.py` → I/O helpers.
  - `preprocess.py` → normalization of raw audio.
  - `features.py` → feature extraction (mel, f0, loudness, harmonic/noise decomposition).
  - `validate.py` → checks dataset consistency with prompts.
  - `inspect_npz.py` → inspects `.npz` feature files for shapes and value ranges.
  - `consistency_check.py` → cross-check between audio, features, metadata.
  - `smoke_test.py` → minimal run test.
- Successfully ran:
  - **Smoke test** (generation + preprocess).
  - **Consistency check** (metadata ↔ audio ↔ npz).
  - **Validate** (feature integrity).
  - **Inspect NPZ** (manual inspection of mel, f0, vuv, loudness).

### Day 2
- Embedding extraction:
  - Added `configs/embeddings.yaml` with paths for text, audio, and FAISS index.
  - Generated **text embeddings** from `data/meta/prompts.csv`.
  - Generated **audio embeddings** from preprocessed `.wav`.
  - Packed per-file embeddings into consolidated index (`faiss.index`, `ids.npy`).
  - `embeddings_smoke_test.py` → minimal run test for embeddings pipeline.
- Retrieval pipeline:
  - Implemented with `src/retrieval/retrieve.py` and `src/retrieval/pack_embeddings.py`.
  - CLAP model used for joint text/audio embedding space.
  - Verified index rebuild and query system.
- Query tests:
  - Low-level descriptors (e.g., *pitch, decay, sustain, brightness*) worked very well.
  - Mid-level descriptors (e.g., *metallic, wooden, harsh*) moderately successful.
  - High-level/metaphorical descriptors (e.g., *harp-like, guitar-like*) less reliable.
- Known issues solved:
  - Fixed missing paths in config.
  - Added `index_dir` for FAISS storage.
  - Clarified CUDA vs CPU usage fallback.
- Results:
  - Retrieval produces **ranked list of `.audio` files** with similarity scores.
  - Demonstrated consistent mapping between prompts and retrieved sounds.

---

## Repository Structure
awol-audio/
│
├── configs/
│   ├── base.yaml                # main audio configuration
│   └── embeddings.yaml          # embedding/retrieval configuration
│
├── data/
│   ├── meta/                    # metadata
│   │   └── prompts.csv
│   ├── raw/                     # raw audio (.wav)
│   ├── processed/               # preprocessed audio and features
│   │   ├── embeddings/          # per-file embeddings
│   │   ├── index/               # faiss index + ids.npy
│   │   └── npz/                 # extracted features (.npz)
│   └── embeddings/              # packed embeddings
│       ├── audio/               # consolidated audio embeddings
│       └── text/                # consolidated text embeddings
│
├── src/
│   ├── analysis/                # analysis pipeline
│   │   ├── audio_io.py
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── validate.py
│   │   ├── inspect_npz.py
│   │   ├── consistency_check.py
│   │   ├── smoke_test.py
│   │   └── embeddings_smoke_test.py
│   │
│   ├── datasets/                # synthetic dataset generation
│   │   └── make_synthetic_plucks.py
│   │
│   ├── retrieval/               # retrieval pipeline
│   │   ├── retrieve.py
│   │   └── pack_embeddings.py
│   │
│   ├── audio_encoder/           # audio embedding models
│   └── text_encoder/            # text embedding models
│
├── README.md
└── requirements.txt
