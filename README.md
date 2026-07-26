# SEPAGen: Transformer-Based Behavior Anomaly Detection for APP Fraud in SEPA Instant Payments


[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This project builds a **decoder-only autoregressive Transformer** trained exclusively on normal per-account SEPA Instant payment sequences to detect behavioural anomalies as candidates for Authorised Push Payment (APP) fraud investigation.

As of this writing, no public dataset exists for SEPA payment fraud. So we first generate **SynSEPA** — a fully synthetic SEPA Instant Credit Transfer dataset of 1.84 million transactions across 10,000 accounts, based on reports published by EBA-ECB 2025 fraud statistics.

The Transformer is trained to predict the next transaction given 20 past transactions. Transactions with low sequence likelihood are flagged as anomalous.

**Core hypothesis:** A generative model trained on normal behaviour learns the temporal nature of individual accounts better than feature-based anomaly detectors, making behavioural deviations (the primary signal in APP fraud) easier to surface.

---

## Results

| Model | AUROC | PR-AUC |
|---|---|---|
| Isolation Forest (baseline) | 0.9514 | 0.1368 |
| SEPAGen Transformer | *see report* | *see report* |

Threshold calibrated at 1% FPR.

---

## Requirements

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

Recommended: Google Colab with T4 GPU for model training (`phase4_transformer.py`).
All other scripts run on CPU.

---

## Quickstart

### Step 0 — Get the data

See `data/README.md` for two options:
- **Download** pre-generated CSVs from the [SynSEPA dataset repository](#)
- **Regenerate** from scratch (10–15 min)

Place `accounts.csv` and `synsep_full_dataset.csv` in the repo root before proceeding.

### Step 1 — Tokenise the dataset

```bash
python src/preprocessing/phase3_step1_tokenise.py
```

### Step 2 — Build sliding-window sequences

```bash
python src/preprocessing/phase3_step2_sequences.py
```

### Step 3 — Run the Isolation Forest baseline (Optional)

```bash
python src/model/sample_baseline.py
```

### Step 4 — Train the Transformer (GPU recommended)

```bash
python src/model/phase4_transformer.py
```

For a quick architecture verification without data:

```bash
python src/model/phase4_transformer.py --test
```

---

## Inference

After training is complete, use `inference_demo.py` to score a new incoming transaction against a 20-transaction account history.

### Prerequisites

The following files must exist before running inference:

- `sepagen_model.pt` — trained model checkpoint (produced by Step 4)
- `data/tokeniser.json` — tokeniser config (produced by Step 1)

### Run the demo

```bash
python src/model/inference_demo.py
```

The script:
1. Loads the saved model checkpoint and tokeniser.
2. Constructs a synthetic history of 20 normal transactions (a retiree making small domestic grocery/utility payments).
3. Scores an anomalous incoming transaction (€8 500, cross-border, new IBAN, 11 pm, 30 min after the previous payment).
4. Prints the anomaly score and flags it as `FLAGGED` or `normal` against a threshold calibrated at 1% FPR.

Example output:

```
Model loaded  (val_loss=0.XXXX)

Account history:   €45–95 grocery/utility, domestic, weekdays 10am
Incoming txn:      €8500, cross-border, new IBAN, 11pm, 30min gap

Anomaly score:     -X.XXXX  (threshold: -4.0268)
Decision:          FLAGGED
```

---
## Dataset

The SynSEPA dataset will be published separately for research use:

- **SynSEPA on Hugging Face:** [link]
- **SynSEPA on Kaggle:** [link]

---

## License

Code: MIT License.
SynSEPA dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
