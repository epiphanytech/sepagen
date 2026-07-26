# SEPAGen: Project Reference Document

> **Course:** Mathematics Behind Generative Models
> **Type:** Solo Capstone Project
> **Timeline:** 7 Days (submission deadline)
> **Version:** 2.0 — Revised after supervisor feedback
> **Status:** Phases 1–4 complete, Phases 5–8 remaining

---

## 📌 Project Name

**SEPAGen: A Decoder-Only Autoregressive Transformer for Behavioural Anomaly Detection in SEPA Instant Payments**

> **Naming note (v2):** Renamed from "GPT-style Transformer" to "decoder-only autoregressive Transformer" — technically precise terminology per supervisor feedback. "GPT" specifically refers to OpenAI's model family; our architecture uses the same underlying mechanism (causal self-attention, autoregressive prediction) without being a GPT model.

---

## ⚠️ Important Framing Correction (v2)

**This project builds a behavioural anomaly detector, not a fraud classifier.**

This distinction matters and is stated explicitly throughout this document and the final report:

- The model learns the probability distribution of **normal** SEPA payment sequences per account
- At inference, it flags transactions with **low likelihood** under that learned distribution
- A flagged transaction is a **candidate for human fraud investigation** — not a confirmed fraud verdict
- Not every anomaly is fraud (e.g. a genuine lottery win, a one-off large legitimate purchase) — and not every fraud is a large statistical outlier (e.g. invoice fraud closely mimics normal payment amounts)
- SynSEPA's APP fraud typologies are used to **construct realistic anomalous transactions for evaluation** — they are not fraud categories the model is trained to recognise or classify

This reframing does not change the architecture or the dataset — it changes how we describe what the system does, and it shapes the evaluation and discussion sections of the final report.

---

## 🎯 Problem Statement

SEPA Instant Credit Transfers settle in under 10 seconds and are irreversible. Authorised Push Payment (APP) fraud — where victims are psychologically manipulated into willingly sending money to fraudster-controlled accounts — is the dominant and fastest growing fraud type in this space, with EU fraud losses hitting **€2.5 billion in 2024 (up 24% YoY)**.

APP fraud is difficult for traditional rule-based systems because the payment is technically legitimate — the sender's identity is verified, authentication is passed, and the instruction is genuine. The strongest available signal is **behavioural deviation** from the sender's normal payment patterns.

This project addresses two compounding challenges unique to SEPA APP fraud detection:

1. **Absence of fraud labels at inference time** — APP fraud looks like a legitimate transaction until it is too late
2. **Extreme scarcity of labelled fraud samples** — no public SEPA fraud dataset exists for training supervised models

---

## 💡 Core Hypothesis

> A decoder-only autoregressive Transformer, trained exclusively on normal per-account SEPA payment sequences, can surface behaviourally anomalous transactions — measured as low sequence likelihood — as effective candidates for APP fraud investigation, within the latency constraints of the SEPA Instant settlement window.

---

## 🏗️ What We Are Building — 7-Day Core Scope

To fit a realistic 7-day solo timeline, the project scope is deliberately narrowed to four components. Everything else considered earlier (CVAE synthesis, multi-model comparison, SHAP, dataset publication) is explicitly **out of scope** for this submission and noted as future work.

### Component 1 — SynSEPA Dataset ✅ Done
Synthetic SEPA Instant transaction dataset — 1.84M transactions, 10,000 accounts, 4 personas, 4 APP fraud typologies. Fully generated and validated (see Phase 2).

### Component 2 — Detection Model: Decoder-Only Autoregressive Transformer ✅ Done
Trained on sequences of normal SEPA transactions, **strictly per account** (sequences are never constructed across different accounts — each sequence is one account's own chronological history). The model learns the conditional probability of each account's own normal payment behaviour. At inference, a new transaction is scored by its **log-likelihood** given that account's own recent history — low likelihood flags it as behaviourally anomalous.

### Component 3 — Baseline: Isolation Forest ✅ Done (kept in scope)
Already implemented and tested (`sample_baseline.py`), achieving AUROC 0.9514 on held-out test data. Kept in scope since it is complete, adds no further development time, and materially strengthens the evaluation section by giving the Transformer a concrete baseline to beat.

### Component 4 — Evaluation ⬜ Core remaining work
Rigorous comparison of the Transformer against the Isolation Forest baseline using a pre-registered metric set and an explicit threshold calibration protocol (defined below). This is the primary remaining scope item.

### Explicitly Out of Scope for This Submission
- LSTM Autoencoder baseline
- CVAE / Diffusion-based synthetic fraud augmentation
- SHAP-based explainability
- Sequence-length ablation study
- Kaggle / HuggingFace dataset publication
- Multi-typology fraud classification

These are listed in the final report's **Future Work** section, not silently dropped — this shows the marker the scope was a deliberate, reasoned decision rather than an oversight.

---

## 🔍 Interpretability, Not Explainability (v2 correction)

Attention weight visualisation is retained as a lightweight interpretability aid, but is **explicitly not presented as a formal model explanation**. Per Jain & Wallace (2019, *"Attention is not Explanation"*), high attention weight on a past transaction indicates the model's internal focus, not a proven causal reason for the anomaly score. In the final report this is described as an **attention-based interpretability signal** intended to help a human analyst triage a flagged transaction faster — not as ground-truth reasoning.

Given the 7-day scope, only a minimal attention visualisation is produced (one or two illustrative examples) — a full explainability module (Phase 7 in the original plan) is descoped.

---

## 🏛️ System Architecture

```
SEPA Transaction History — STRICTLY PER ACCOUNT
[Sender IBAN | Beneficiary IBAN | Amount | Timestamp | Reference | Payment Type]
(sequences are constructed independently for each account_id —
 never mixed across different customers)
                            ↓
                   Tokenisation Layer
     (derived/bucketed features only — see IBAN note below)
                            ↓
        Decoder-Only Autoregressive Transformer
           (trained on normal transactions only)
     (learns each account's own conditional distribution
      over its next transaction, given its own history)
                            ↓
                 Log-Likelihood Scoring
          (how probable is this transaction given
           THIS account's own recent history?)
                            ↓
              Threshold Calibration (see protocol below)
                ┌─────────────────────────────┐
                │  High likelihood → ✅ Normal          │
                │  Low likelihood  → 🚩 Behavioural       │
                │                    anomaly — flagged     │
                │                    for review, NOT a      │
                │                    confirmed fraud verdict │
                └─────────────────────────────┘
                            ↓
        Attention Weight Visualisation (interpretability
        aid only — illustrative, not formal explanation)
```

### Why raw IBANs are never tokenised (v2 addition)

High-cardinality fields such as IBAN strings are deliberately **not** fed to the model as tokens. Directly tokenising millions of near-unique IBAN strings would let the model memorise specific account numbers instead of learning general behavioural patterns, and would fail to generalise to any IBAN not seen during training. Instead, IBANs are reduced to **derived, low-cardinality features** before tokenisation:

| Raw field | Derived feature used instead | Cardinality |
|---|---|---|
| `beneficiary_iban` (raw string) | `is_new_beneficiary` (0/1) | 2 |
| `beneficiary_iban` country prefix | `country_type` (domestic / eu / non-eu) | 3 |

The raw IBAN string is retained in the CSV only as a human-readable reference column — it is never part of the token vocabulary or model input.

---

## 📦 Dataset: SynSEPA

**Name:** SynSEPA — A Synthetic SEPA Instant Payment Dataset for Behavioural Anomaly Detection Research

### What it contains
- 1,832,448 normal SEPA Instant Credit Transfer transactions, sequenced per account
- 7,112 APP fraud transactions across 4 typologies, injected as realistic behavioural anomalies:
  - Bank / authority impersonation
  - Invoice / mandate fraud
  - Romance scam
  - CEO / business email compromise (BEC) fraud

### How it is generated (v2 clarification)
- **Normal transactions:** rule-based persona simulation (employee / student / retiree / business), calibrated against EBA-ECB 2025 fraud report statistics (fraud rate, cross-border rate, transaction volume patterns)
- **Fraud transactions:** generated by **rule-based scripts that encode the documented behavioural signatures** from the EPC 2025 Payment Threats and Fraud Trends Report for each typology (e.g. impersonation → large amount + new domestic IBAN + urgent remittance text + short delay after a normal transaction). **No LLM is used to generate fraud transactions in the current dataset** — every fraud sample is deterministically constructed from an explicit, documented rule set, which is fully reproducible from the provided generation scripts.
- Validated against EBA-ECB published aggregate statistics — 41 of 42 automated validation checks passed (see `validation_report.txt`)

### Publication (descoped for this submission)
Kaggle / HuggingFace publication was part of the original plan but is **out of scope for the 7-day submission**. The dataset, generation code, and datasheet already exist and remain publishable as future work.

---

## 🧮 Threshold Calibration Protocol (v2 addition — was previously undefined)

The anomaly threshold is not chosen arbitrarily. It is calibrated on the validation split and then frozen before touching the test split:

1. Score every transaction in the **validation set** (October 2024, contains a small number of fraud cases) using the trained model's log-likelihood
2. Build the ROC curve from validation scores against the `is_fraud` validation labels
3. Select the score threshold corresponding to a **1% false positive rate** — chosen as an operationally realistic rate for a bank's fraud analyst review queue
4. Report results at additional operating points (0.5%, 2%, 5% FPR) for robustness, since the "right" FPR is a business decision, not a fixed constant
5. Apply the **frozen validation-derived threshold** to the held-out test set (November–December 2024) to produce the final reported Precision, Recall and F1

This mirrors the protocol already implemented in `sample_baseline.py` and `phase4_transformer.py` — it now needs to be written up explicitly in the report rather than left implicit in the code.

---

## 📐 Evaluation Protocol (v2 addition — was previously incomplete)

| Metric | Purpose |
|---|---|
| **AUROC** | Overall ranking quality — threshold-independent |
| **PR-AUC** | Primary metric given ~0.4% fraud rate — AUROC alone is misleadingly optimistic under heavy class imbalance |
| **Precision @ 1% FPR** | Of the transactions flagged, what fraction are true fraud |
| **Recall @ 1% FPR** | Of true fraud cases, what fraction are caught |
| **F1 @ 1% FPR** | Single balanced score at the calibrated operating point |

**Baseline comparison (in scope):** Isolation Forest, trained and evaluated on an identical time-based split, using the same threshold calibration protocol. Isolation Forest currently achieves AUROC 0.9514 / PR-AUC 0.1368 on the test set — the Transformer's job is to improve materially on PR-AUC by using sequential context that Isolation Forest cannot see.

**Baseline comparison (out of scope):** LSTM Autoencoder — noted explicitly as future work rather than silently omitted.

---

## 📐 Mathematical Coverage (Course Syllabus Map)

| Course Topic | Where It Appears in This Project |
|---|---|
| Decoder-only autoregressive Transformer / Self-Attention | Core detection model architecture |
| Autoregressive Likelihood | Anomaly scoring via log-probability |
| Positional Encoding | Temporal ordering of each account's transaction sequence |
| Causal Masking | Ensures model only uses past transactions at inference |
| VAE / ELBO | Background — contrasted with the autoregressive approach used here |
| Generative vs Discriminative Models | Core framing of the anomaly detection approach |

---

## 📄 Key Papers

| Paper | Why It Matters |
|---|---|
| Vaswani et al. (2017) — *Attention Is All You Need* | Core Transformer architecture |
| Zhao et al. (2023) — *Generative Pretraining at Scale for Fraud Detection* (WeChat Pay) | Closest prior work — same autoregressive-sequence framing, same "anomaly not fraud-type" framing |
| Jain & Wallace (2019) — *Attention Is Not Explanation* | Justifies the interpretability-not-explainability framing of attention visualisation |
| EBA-ECB Payment Fraud Report (2025) | Real-world statistics for dataset calibration |
| EPC Payment Threats and Fraud Trends Report (2025) | Source of the documented fraud typology signatures used in rule-based fraud generation |

---

## 🗓️ Revised 7-Day Plan

Phases 1–4 are already complete from the earlier, longer timeline. The remaining 7 days focus on evaluation, a light interpretability pass, and write-up.

---

### ✅ Phase 1 — Understanding (Complete)
Papers read, SEPA structure understood, APP fraud typologies understood, architecture sketched.

---

### ✅ Phase 2 — SynSEPA Dataset (Complete)
10,000 accounts, 1.84M normal transactions, 7,112 rule-based fraud transactions, validated (41/42 checks).

---

### ✅ Phase 3 — Data Preparation (Complete)
Tokenisation (66-token vocabulary, no raw IBANs), per-account sliding-window sequences, time-based train/val/test split.

---

### ✅ Phase 4 — Transformer Model (Complete)
Decoder-only autoregressive Transformer implemented and architecture-verified (~102K parameters). Ready to train in Colab.

---

### ⬜ Day 1–2 — Train the Transformer
**Tasks:**
- [ ] Run training in Google Colab (T4 GPU) — approx. 20–30 minutes for 10 epochs
- [ ] Monitor train/validation loss curves, confirm no divergence or severe overfitting
- [ ] Save best checkpoint by validation loss

**Output:** Trained model checkpoint (`sepagen_model.pt`) + training loss curve

---

### ⬜ Day 3 — Threshold Calibration & Evaluation
**Tasks:**
- [ ] Apply the threshold calibration protocol on the validation split
- [ ] Compute AUROC, PR-AUC, Precision/Recall/F1 @ 1% FPR on the test split
- [ ] Re-run the existing Isolation Forest baseline on the identical split for a fair side-by-side comparison
- [ ] Produce ROC and Precision-Recall curve plots for both models

**Output:** Results table (Transformer vs Isolation Forest) + ROC/PR curve plots

---

### ⬜ Day 4 — Light Interpretability Pass
**Tasks:**
- [ ] Extract attention weights for 2–3 illustrative flagged transactions (one true positive, one false positive if available)
- [ ] Produce simple attention heatmap visualisations
- [ ] Write the interpretability caveat paragraph (attention ≠ explanation) into the report draft

**Output:** 2–3 attention heatmaps + short qualitative discussion

---

### ⬜ Day 5–6 — Write-Up
**Report structure:**
1. Abstract
2. Introduction — problem, motivation, and the anomaly-vs-fraud framing
3. Related Work — WeChat Pay paper, Jain & Wallace on attention
4. Dataset — SynSEPA generation methodology, rule-based (not LLM-based) fraud construction, validation results
5. Methodology — architecture, per-account sequencing, no-raw-IBAN tokenisation rationale, threshold calibration protocol
6. Experiments — evaluation protocol, Transformer vs Isolation Forest results
7. Interpretability — attention visualisation with explicit caveats
8. Limitations & Future Work — LSTM baseline, CVAE synthesis, dataset publication, ablations, all explicitly listed as descoped
9. Conclusion
10. References

**Output:** Draft report, all sections complete

---

### ⬜ Day 7 — Review & Submit
**Tasks:**
- [ ] Proofread for the terminology corrections throughout (anomaly vs fraud, decoder-only vs GPT-style, interpretability vs explainability)
- [ ] Confirm every claim in the report is backed by a number, a citation, or an explicit limitation statement
- [ ] Final packaging — code, report, dataset sample

**Output:** Final submission

---

## 🔑 Key Principles to Remember

**1. Most of the supervisor's feedback is a writing problem, not a code problem**
The per-account sequencing, the rule-based (non-LLM) fraud generation, and the derived-feature (non-raw-IBAN) tokenisation were already implemented correctly — they simply weren't stated explicitly enough in the original proposal. Say it clearly in the report rather than re-engineering something that already works.

**2. Scope discipline over scope creep**
Everything not in the four core components (Dataset, Transformer, Isolation Forest, Evaluation) goes in Future Work, not into this week's task list.

**3. Anomaly detection, never "fraud detection," in the report's own voice**
Consistent terminology throughout protects the project from the single most substantive piece of feedback received.

**4. Threshold and evaluation protocol must appear explicitly in the report**
The logic already exists in code — Day 3 is about writing it up rigorously, not inventing something new.

---

## 🧰 Tools and Libraries

| Tool | Purpose |
|---|---|
| Python | Primary language |
| PyTorch | Transformer implementation |
| scikit-learn | Isolation Forest baseline, evaluation metrics |
| pandas / numpy | Data processing |
| matplotlib | ROC/PR curves, attention heatmaps |
| Google Colab (T4 GPU) | Model training |

---

## 📊 Success Criteria (Revised for 7-Day Scope)

The project is successful if:

- [x] SynSEPA dataset is generated and validated
- [ ] Transformer trains without divergence and produces a saved checkpoint
- [ ] Evaluation protocol (metrics + threshold calibration) is fully documented and executed
- [ ] Transformer is compared against the Isolation Forest baseline on an identical test split
- [ ] Report uses corrected terminology consistently (anomaly detection, decoder-only Transformer, interpretability)
- [ ] Report explicitly lists descoped items as Future Work rather than omitting them silently

---

*This document is Version 2 — revised after supervisor feedback to correct terminology, add the previously missing threshold calibration and evaluation protocols, and tighten scope to a realistic 7-day solo submission.*
