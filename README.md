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

Threshold calibrated at 1% FPR (operationally realistic for a bank alert queue).

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

## APP Fraud Typologies (EPC 2025)

| Typology | Victims | Signature |
|---|---|---|
| Bank/Authority Impersonation | 1,440 | Single large transfer, new domestic IBAN, "safe account" text |
| Invoice / Mandate Fraud | 900 | Amount mirrors normal supplier payments, business accounts only |
| Romance Scam | 720 | 4–8 escalating payments to same foreign IBAN over weeks |
| CEO / BEC Fraud | 540 | Large amount, non-EU IBAN, Friday afternoon, business accounts |

---

## Dataset

The SynSEPA dataset will be published separately for research use:

- **SynSEPA on Hugging Face:** [link]
- **SynSEPA on Kaggle:** [link]

---

## License

Code: MIT License.
SynSEPA dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
