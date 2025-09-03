# AWOL-Audio Text-to-Synth Pipeline

This repository implements a **text-to-synthesis** pipeline that maps natural-language prompts (e.g., “high pitched wooden pluck, long sustain”) to **synth parameters** and renders audio.
The design follows the **AWOL** philosophy: few examples, fast training, and a simple, modular stack you can re-run and inspect end-to-end.

At a high level:

1. **Preprocess** + **features** → compute analysis targets from audio (pitch, decay, brightness, …).

2. **Embeddings** → extract text and audio embeddings (CLAP / Sentence-Transformers for text; CLAP for audio).

3. **ParamReg** (pretrained, kept fixed) → learn to predict normalized synth parameters from audio-embedding space.

4. **Mapper** → learn to map text embeddings → audio embeddings (with an auxiliary loss through ParamReg).

5. **Pipeline** → at inference, a text prompt is encoded, mapped to an audio embedding, converted to synth params, and rendered.

   Optional: build a FAISS index for similarity/retrieval diagnostics (R@k) and/or neighbor blending (if enabled in configs/CLI).
---

## Dataset 
36 audio examples, recorded manually on acoustic guitar (single plucks) + 36 text prompts, one per audio.

Prompts cover a diverse set of attributes: high/low pitch, long/short sustain, neck/bridge, wooden/metallic, bright/crisp/dull, etc.
This variety intentionally supports the few-shot setup (AWOL ethos).

A first cleaning pass was done manually (trimming, coarse quality control). No automatic onset-trim tool is used in this repo.

The dataset is self-produced and is included only as raw audio (no generated outputs are checked in).

Note: the small scale is by design; the model trains quickly and produces acceptable audio for the majority of prompts, with known limitations (e.g., some “short sustain” sounding “long”). Those are discussed in the report; the README focuses on pipeline usage and reproducibility.

## Pipeline Modules (what each script does)

### 1) Analysis 
  - `src/analysis/preprocess.py`
          Loads raw WAVs, harmonizes sample rate/mono, organizes splits/IDs, and prepares paths for downstream steps.

  - `src/analysis/features.py`
          Computes analysis targets from .npz intermediates (pitch in Hz, T60/decay, brightness, damping, pick position, …).
          These targets are used to supervise ParamReg training and to evaluate predictions in real units.

  - `src/text_encoder/extract_text_embeddings.py`
          Encodes all text prompts into text embeddings (e.g., Sentence-Transformers/CLAP text head) and writes *.text.npy.

  - `src/audio_encoder/extract_audio_embeddings.py`
          Encodes all audio into audio embeddings (e.g., CLAP audio head) and writes *.audio.npy.

  Configs: `configs/base.yaml` (preprocess + features), `configs/embeddings.yaml` (text/audio encoders).

### 2) Retrieval (optional but useful)
  - `scripts/build_faiss_index.py`
          Packs L2-normalized audio embeddings into a FAISS (Facebook AI Similarity Index) Inner-Product (cosine) index + names.npy.
          Used by evaluation scripts to compute retrieval metrics (R@1/5/10) and (optionally) for neighbor blending in the pipeline.

  Config: uses the same embedding output dirs as `configs/embeddings.yaml`.

### 3) Synth / Parameter Regressor
   - `src/synth/train_params.py`
          Trains ParamReg: a regressor from audio embedding → normalized synth parameters [0..1] (per spec).
          Normalization ranges/scales (e.g., log_hz for pitch) come from configs/params.yaml.

  - `src/synth/evaluate_params.py`
          Loads a checkpoint and reports MAE (Mean Absolute Error) per parameter in real units (e.g., Hz for pitch, seconds for T60).
          Useful to verify parameterization quality.

  (Internals in this folder)
  `param_regressor.py`, `targets_from_npz.py`, `predict_params.py` — the first two are the core model & target extraction, the last one is a inference-only utility;

  Config: `configs/params.yaml` (audio section + synth.params specs + model hyper-params).)

### 4) Mapper (Text -> Audio-Embedding)
  - `src/mapper/train_mapper.py`
          Trains an MLP to map text embeddings -> audio embeddings.
          Loss = embedding loss (cosine/MSE) + auxiliary param loss: predicted audio embeddings are fed into the frozen ParamReg, and the resulting parameters are compared to ground-truth targets [0..1].
          This encourages the mapper to land in semantically meaningful regions of the audio-embedding space.

  - `src/mapper/evaluate_mapper.py`
          Reports mean cosine between predicted and true audio embeddings on the validation split, and R@k if a FAISS index is present.
          Also runs the frozen ParamReg on predicted embeddings and prints per-parameter MAE in real units.

  Config: `configs/mapper.yaml` (model size, losses, λ weights, paths to embeddings/npz and to the ParamReg checkpoint).

### 5) Pipeline (Text -> Audio)
  - `src/pipeline/text2synth.py`
          End-to-end inference from a prompt:
          prompt -> text emb -> mapper -> audio emb -> ParamReg -> synth params -> render WAV.
          Optionally, can be disabled neighbor blending and set topk=0.

  **Renderer**: a minimal Karplus–Strong plucked-string synthesizer (fast, lightweight), purpose-built for pluck-like sounds. It’s used to audibilize the predicted parameters end-to-end; full synthesis details and trade-offs are discussed in the report.
          

  - `src/pipeline/batch_text2synth.py`
          Batch version that reads a CSV of prompts (e.g., tests/prompts_text2synth.csv).

  - `src/pipeline/collect_text2synth_csv.py`
          Helper to build CSVs of prompts for repeated experiments.

      Config: `configs/pipeline.yaml` (paths to checkpoints, output dir, render settings, optional retrieval).

### 6) Diagnostics
  These scripts do not belong to the strict training pipeline, but are useful for quality control:

  - `src/analysis/scan_npz_dataset.py`
          Scans .npz and writes a CSV report (range checks, voiced ratio, late peaks, brightness thresholds, etc.).
          Used to verify dataset health before training.

  - `src/retrieval/retrieve.py`, `src/retrieval/batch_free_query.py`
          Utilities to query the FAISS index by text or embedding to inspect nearest neighbors.
      
  - `src/analysis/validate.py`, `src/analysis/scan_npz_dataset.py`
          Assorted checks/visualizations

  - `scripts/analyze_generated_audio.py`
          Summaries/statistics over rendered audio (e.g., distributions vs. targets).

      These tools are not required to run the main demo, but they help document/justify behavior in the report.

## Reproducibility 
  ### Presequisites
  - Python: tested on version 3.8 and 3.10. Works on both versions. 
  - Create and activate a virtual environment 
  - Install dependencies: 
      `pip install -r requirements.txt`
  - Run from the repository root
  
  ### Data and artifacts present in the repo
  - Raw data: included (data/raw/).
  - Checkpoints: included (checkpoints/paramreg/paramreg_best.pt, checkpoints/mapper/mapper_best.pt).
  - Summaries/reports: included (e.g., reports/…).
  - Generated outputs: not included (in .gitignore); they will be re-created under outputs/.

  ### Commands
  1) **Preprocess**
      `python -m src.analysis.preprocess --config configs/base.yaml`
  
  2) **Features**
      `python -m src.analysis.features --config configs/base.yaml`

  3) **Audio Embeddings**
      `python -m src.audio_encoder.extract_audio_embeddings --config configs/embeddings.yaml`
  
  4) **Text Embeddings** 
      `python -m src.text_encoder.extract_text_embeddings --config configs/embeddings.yaml`
  
  5) **Build FAISS index build** (required for evaluation):
      `python scripts/build_faiss_index.py` 
     
  6) **Train ParamReg** (optional: the provided checkpoint can be used)
      `python -m src.synth.train_params --config configs/params.yaml`
  
  7) **Evaluate ParamReg**
      `python -m src.synth.evaluate_params --config configs/params.yaml --ckpt checkpoints/paramreg/paramreg_best.pt`
  
  8) **Train Mapper** (optional: the provided checkpoint can be used)
      `python -m src.mapper.train_mapper --config configs/mapper.yaml`

  9) **Evaluate Mapper**
      `python -m src.mapper.evaluate_mapper --config configs/mapper.yaml --ckpt checkpoints/mapper/mapper_best.pt`
  
  10) **Single-prompt inference (Text -> Synth)**
      `python -m src.pipeline.text2synth --config configs/pipeline.yaml --query "low pitched pluck with long sustain" --topk 0`
  
  11) **Batch prompt**
      `python -m src.pipeline.batch_text2synth --config configs/pipeline.yaml --csv tests/prompts_text2synth.csv --no-neigh`

**Optional: Execution with retrieval**
- Retrieval execution (top-k > 0)
     e.g. k = 5
      `python -m src.pipeline.text2synth --config configs/pipeline.yaml --query "low pitched pluck with long sustain" --topk 5`

What happens: the system takes the embedding predicted by the mapper and retrieves the 5 most similar audios from the FAISS index; then combines them (usually average/weighted-average, according to your internal settings) before passing the result to the ParamReg -> summary parameters -> WAV.

- Variant with batch retrieval (remove --no-neigh)
      `python -m src.pipeline.batch_text2synth --config configs/pipeline.yaml --csv tests/prompts_text2synth.csv`
  
**Optional: Health check**
-  Writes a dataset scan report
      `python -m src.analysis.scan_npz_dataset --config configs/base.yaml --out_csv reports/dataset/npz_scan.csv`

- Collects all text2synth JSONs under outputs and writes a single CSV (JSON path, WAV path, prompt, and predicted synth parameters)
      `python -m src.pipeline.collect_text2synth_csv --json_dir outputs --out_csv reports/summary_text2synth.csv --prompts_csv tests/prompts_text2synth.csv`
  

  ## Known limitations 
    A few prompts (especially short vs. long sustain) may not perfectly match perceived outcomes.
    This stems from the tiny dataset and the simplicity of the parameterization—intentional trade-offs for an AWOL-style, reproducible, fast-to-train system. 
    The approach remains coherent: text controls a compact param set, the model learns from few samples and remains explainable.

  


  




      





