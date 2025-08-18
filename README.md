# AWOL-Audio (Text-to-Audio Prototype)

This repository implements an **AWOL-inspired pipeline** for text-to-audio generation, starting with synthetic data and progressively extending to real datasets.

## Current Status (Day 1)
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
- Successfully ran:
  - **Smoke test** (generation + preprocess).
  - **Consistency check** (metadata ↔ audio ↔ npz).
  - **Validate** (feature integrity).
  - **Inspect NPZ** (manual inspection of mel, f0, vuv, loudness).

## Repository Structure
awol-audio/
│
├── configs/
│   └── base.yaml                # main configuration file
│
├── data/
│   ├── meta/                    # metadata
│   │   └── prompts.csv
│   ├── raw/                     # raw audio (.wav)
│   ├── processed/               # preprocessed audio (.wav)
│   └── npz/                     # extracted features (.npz)
│
├── src/
│   ├── analysis/                # analysis pipeline
│   │   ├── audio_io.py
│   │   ├── preprocess.py
│   │   ├── features.py
│   │   ├── validate.py
│   │   ├── inspect_npz.py
│   │   ├── consistency_check.py
│   │   └── smoke_test.py
│   │
│   └── datasets/                # synthetic dataset generation
│       └── make_synthetic_plucks.py
│
├── README.md
└── requirements.txt

