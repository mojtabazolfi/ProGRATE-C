# ProGRATE‑C: Profiling Graph-Residual Adaptive Transformer Ensemble for Cancer diagnostics based on cfDNA 

<p align="center">
  <img src="docs/architecture.png" alt="ProGRATE-C architecture" width="900"/>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-research--use--only-orange.svg"></a>
  <a href="https://doi.org/"><img alt="DOI" src="https://img.shields.io/badge/DOI-pending-lightgrey.svg"></a>
</p>

**ProGRATE‑C** is an adaptive multi‑view deep‑learning framework for **non‑invasive cancer detection from cell‑free DNA (cfDNA) end‑motif profiles**. It integrates two complementary representational views of the same fragmentation signal:

* **MLET — Motif Language Embedding Transformer**:
  a dual‑view transformer that models the ranked motif sequence **both** as an *ordered sequence* and as an *unordered set*.
* **BiG — Batch‑inductive Dynamic Graph**:
  a BiLSTM semantic encoder coupled with a per‑batch, featureless heterogeneous GCN over `doc–motif`, `motif–motif (PMI)`, and `motif–label (TF‑CRF)` edges.

The two views are merged through **hierarchical adaptive gating + cross‑attention fusion**, allowing the model to weight each representation on a *per‑sample* basis and preventing branch collapse through temperature annealing.

The repository contains the full pipeline: data parsing, cached DNABERT‑4mer per‑motif embeddings, 10‑fold stratified cross‑validation, training, evaluation, publication‑quality figure generation, calibration/robustness analyses, and reproducibility reports.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Citation](#citation)
6. [License](#license)

---

## Highlights

- 🧬 **Dual‑view motif modelling** — captures both order‑aware (sequence transformer) and order‑invariant (set transformer) information from the same 4‑mer embedding matrix.
- 🕸️ **Batch‑inductive graph branch** — rebuilds only the doc‑side edges each batch; the `motif–motif` (PMI) and `motif–label` (TF‑CRF) statistics are computed **once from the training split** and reused as frozen buffers. **No test‑set leakage.**
- ⚖️ **Leakage‑controlled protocol** — vocabulary, IDF, PMI, and TF‑CRF statistics are recomputed per fold from the fold's training subset only.
- 🔀 **Hierarchical adaptive fusion** — two temperature‑annealed gating layers (5.0 → 1.0 over 30 epochs) prevent branch collapse and enable sample‑specific routing.
- 🧊 **Frozen DNABERT‑4mer** — 768‑dim per‑motif embeddings are cached to `float16 .npy` for fast reuse across folds and runs.
- 📊 **Publication‑ready reporting** — per‑fold CSV/XLSX, mean ± std ROC curves, loss/accuracy curves, calibration, SHAP‑style feature attribution, and JSON run reports.
- 💻 **Small footprint** — the full model (~8.5 M parameters) trains on a **single consumer‑grade GPU** with 10‑fold CV in under an hour on the multi‑cancer cohort.

---

## Repository Structure

```
prograte-c/
├── config.py                 # Central configuration (paths, hyperparameters, device)
├── dataloader_fusion.py      # TSV parser, vocab, PMI/IDF/TF-CRF stats, CV folds
├── embeddings.py             # Frozen DNABERT-4mer per-motif embedding extractor (+ cache)
├── model_fusion.py           # MLET + BiG + hierarchical adaptive fusion
├── train_fusion.py           # Main entry point — 10-fold stratified CV training
├── utils.py                  # Seeding, metrics, focal loss, schedulers, early stopping
├── reporting.py              # Parameter counts, inference benchmark, JSON/Excel reports
├── cv_report.py              # CSV/Excel exports + Q1-style loss/acc/ROC figures
├── generate_plots.py         # Regenerate figures from a saved epoch-metrics CSV
└── requirements.txt
```

---

## Installation

Tested on **Python 3.10.15**, **PyTorch 2.13.0**, **CUDA 12.x**, and works on CPU / MPS too.

```bash
# 1. Clone
git clone https://github.com/mojtabazolfi/ProGRATE-C.git
cd ProGRATE-C

# 2. (Recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`:

```txt
torch>=2.0
transformers>=4.40
scikit-learn>=1.3
numpy>=1.24
pandas
matplotlib
tqdm>=4.65
openpyxl>=3.1         # required for .xlsx reports
```

> The first run downloads the `Taykhoom/DNABERT-4mer` weights from the Hugging Face Hub (~400 MB). Internet is required **once**.


---

## Quick Start

### 1. Full 10‑fold CV training (default)

```bash
python train_fusion.py
```

### 2. Quick sanity check (~3 minutes)

```bash
python train_fusion.py --test-mode
```

This runs 3 folds × 3 epochs so you can verify the environment before committing to a full run.

### 3. Force‑rebuild the DNABERT embedding cache

```bash
python train_fusion.py --force-embeddings
```

### 4. Regenerate publication figures from a saved CSV

```bash
python generate_plots.py
```

---

## Citation


---

## License

This project is released under the **MIT License** — see `LICENSE` for details.
**Research use only.** Not intended for clinical diagnosis.

---

## Contact

**Mojtaba Zolfi** — Department of Electrical and Computer Engineering,
University of Science and Technology of Mazandaran, Behshahr, Iran
📧 *mojtaba.zolfi.244@gmail.com*
🔗 [github.com/mojtabazolif](https://github.com/mojtabazolif)
