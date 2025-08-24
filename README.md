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

### Day 3 — Supervised Mapper (Text → Audio Embeddings)
**Objective:**  
1. Create a supervised **Mapper** from text embeddings → audio embeddings (CLAP space).  
2. Train on synthetic pluck pairs (text/audio aligned by filename).  
3. Evaluate with cosine similarity and indirect retrieval (predict → search in FAISS).  

**Steps implemented:**
- **Setup**  
  - Added `configs/mapper.yaml` with device, paths, model, training, and evaluation settings.  
  - Installed `scikit-learn` for train/validation split.  
  - Created checkpoint directory (`checkpoints/mapper`).  

- **Model**  
  - Implemented `src/mapper/mapper_mlp.py`: a simple **MLP** mapping text embeddings → audio embeddings.  
  - Supports multiple hidden layers, dropout, and optional L2 normalization of outputs.  

- **Training**  
  - `src/mapper/train_mapper.py` handles supervised training with cosine or MSE loss, AdamW optimizer, and early stopping.  
  - Uses paired embeddings (`.text.npy` / `.audio.npy`) from Day 2.  
  - Produces checkpoint `mapper_best.pt` with best validation loss.  

- **Evaluation**  
  - `src/mapper/evaluate_mapper.py` computes:  
    - Mean **cosine similarity** on validation pairs.  
    - **Retrieval metrics**: Recall@1 and Recall@5 via FAISS index.  

- **Interactive usage**  
  - `src/mapper/predict_and_retrieve.py`:  
    - Loads a text embedding (e.g., `synthetic_pluck_012.text.npy`).  
    - Predicts audio embedding with Mapper.  
    - Retrieves top-K nearest neighbors from FAISS index.  

- **Smoke test**  
  - `src/mapper/mapper_smoke.py`: quick 2-epoch test to check training loop and decreasing loss.  

**Expected results:**  
- A trained **mapper checkpoint** (`checkpoints/mapper/mapper_best.pt`).  
- Validation metrics: cosine similarity + R@1, R@5 retrieval scores.  
- Verified end-to-end flow: **text embedding → Mapper → predicted audio embedding → FAISS retrieval**.  


### Day 4 — Free Query & Prompt Fusion
**Objective:** Extend Mapper usage to handle free-text queries and prompt combinations.  

**New features implemented:**
- **Free Query Prediction**  
  - `src/mapper/predict_free_query.py` allows predicting audio embeddings directly from **unseen textual prompts** (not limited to `prompts.csv`).  
  - Example run:  
    ```
    python -m src.mapper.predict_free_query --config configs/mapper.yaml --ckpt checkpoints/mapper/mapper_best.pt --prompt "warm wooden pluck with short sustain" --topk 5
    ```  
  - Output: ranked list of nearest audio files, confirming generalization of Mapper.  

- **Prompt Fusion**  
  - `src/tools/prompt_fusion.py` merges multiple prompts into a **blended embedding** before retrieval.  
  - Example run:  
    ```
    python -m src.tools.prompt_fusion --config configs/mapper.yaml --ckpt checkpoints/mapper/mapper_best.pt --prompts "bright pluck" "sparkling pluck" "crisp short-decay tone" --topk 5
    ```  
  - Produces audio candidates matching **combined descriptors**.  

- **Batch Free Query**  
  - `src/tools/batch_free.py` runs multiple unseen prompts at once (stored in `.txt` or `.csv`).  
  - Results saved into `batch_free.csv` with columns: `prompt, rank, name, score`.  
  - Example output:  
    ```
    bright metallic pluck with long sustain,1,synthetic_pluck_034.audio,0.9839
    dark mellow pluck with short decay,1,synthetic_pluck_031.audio,0.9688
    warm wooden tone with soft attack,1,synthetic_pluck_031.audio,0.9764
    ```
  - Confirms stable retrieval across diverse query styles.  

**Results:**  
- Mapper successfully generalizes to **novel text prompts**.  
- Prompt fusion allows richer, more expressive descriptions.  
- Batch querying enables systematic evaluation across prompt sets.  


### Day 5/6 — Prototype Synthesizer

**Objective:**  
- Implement a **prototype audio synthesizer** driven by prompt-derived parameters.  
- Integrate a **Karplus–Strong decoder** (`synth_decoder.py`) into the pipeline.  
- Allow generation of pluck-like audio signals from free-text prompts.  

**Implementation:**  
- Added `run_synth.py` with rule-based parameter extraction (`rule_params_from_prompt`).  
- Added `KarplusStrongDecoder` to map high-level controls (pitch, decay, brightness, material).  
- Configured paths and defaults in `configs/synth.yaml`.  
- Tested the synthesizer on multiple prompts (e.g., *bright metallic pluck*, *dark mellow pluck*, *high-pitch bright pluck*).  

**Notes:**  
- Current tests show **audible variation** between prompts (mainly pitch/decay).  
- We have **not yet tested the mapper in synthesis** — parameters are currently forced via simple rules for validation.  
- This provides a **working baseline** to verify that the pipeline produces coherent audio before moving to real data and more complex decoders.



## Repository Structure
awol-audio/
│
├── checkpoints/
│   ├── clap/           
│   └── mapper/
|
├── configs/
│   ├── base.yaml                # main audio configuration
|   ├── embeddings.yaml          # embedding/retrieval configuration
|   ├── synth.yaml
│   └── mapper.yaml 
|
│
├── data/
│   ├── meta/           # metadata
|   |   ├── free_prompts.csv
│   │   └── prompts.csv
│   ├── raw/                     # raw audio (.wav)
│   ├── processed/               # preprocessed audio and features
│   │   ├── embeddings/          # per-file embeddings
|   |   |    ├── audio/               
|   │   |    └── text/ 
│   │   ├── index/               # faiss index + ids.npy
│   │   └── npz/                 # extracted features (.npz)
|   ├── results/ 
│   └── embeddings/              # packed embeddings
│       ├── audio.npy               # consolidated audio embeddings
│       └── text.npy                # consolidated text embeddings
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
|   |   ├── spectral_feats.py
│   │   └── embeddings_smoke_test.py
│   │
│   ├── datasets/                # synthetic dataset generation
│   │   └── make_synthetic_plucks.py
│   │
│   ├── retrieval/               # retrieval pipeline
│   │   ├── retrieve.py
|   |   ├── batch_free_query.py
│   │   └── pack_embeddings.py
|   |
|   ├── synth/ 
|   |   ├── run_synth.py
|   |   └── synth_decoder.py
│   │   
|   |
│   ├── audio_encoder/           # audio embedding models
│   ├──text_encoder/            # text embedding models
|   |
|   |── tools/
|   |
|   └── mapper/
│      ├── evaluate_mapper.py
│      ├── mapper_mlp.py
│      ├── mapper_smoke.py
│      ├── predict_and_retrieve.py
|      ├── predict_free_query.py
│      └── train_mapper.py
|   
│
├── README.md
└── requirements.txt
