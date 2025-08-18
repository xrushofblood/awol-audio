# AWOL-Audio 🎵

This repository implements **AWOL (Amusing Wails On Loop) for audio**, inspired by the original paper *Analysis WithOut synthesis using Language* ([arXiv:2404.03042](https://arxiv.org/abs/2404.03042)) and adapted to the project specification of the **Deep Learning and Applied AI (DLAI) course** [Project 9 - AWOL](./project_list.pdf).

The goal is to explore whether the philosophy of AWOL, originally applied to 3D geometry generation from language, can be used to **generate audio from textual prompts**.  
The pipeline involves:
1. Extracting **text embeddings** (via CLAP or similar models).  
2. Mapping them into an **audio parameter space** (f0, loudness, harmonic parameters).  
3. Using a **parametric synthesizer** (DDSP, MelGAN, or alternatives) to reconstruct sound.  
4. Exploring both **retrieval-based** and **generative** approaches, in line with the original AWOL paper.

---

## Repository structure
awol-audio/
data/
raw/ # Original wav files (not tracked by git)
processed/ # Normalized/preprocessed audio
meta/
prompts.csv # Metadata: file_name, prompt
src/
analysis/
audio_io.py
preprocess.py
features.py
pack_npz.py
validate.py
configs/
base.yaml
notebooks/
01_quick_listen.ipynb # Optional listening tests
README.md

