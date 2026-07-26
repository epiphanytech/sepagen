# SEPAGen — Complete Project Summary & Handoff Document

> **Purpose of this document:** This is a complete handoff summary of the SEPAGen capstone project, intended to be given to Claude Code to (1) organize the project into a clean, committable GitHub repository structure, and (2) serve as the source material for generating the final academic report. Everything needed to understand what was built, why, and what remains is captured here.

---

## 1. Project Identity

**Title:** SEPAGen: A Decoder-Only Autoregressive Transformer for Behavioural Anomaly Detection in SEPA Instant Payments

**Course:** Mathematics Behind Generative Models (capstone project)

**Type:** Solo project, 7-day submission timeline (as of the most recent scope revision)

**One-sentence summary:** A synthetic SEPA Instant payment dataset (SynSEPA) was generated from scratch, and a small decoder-only autoregressive Transformer was trained to detect behavioural anomalies in per-account payment sequences as candidates for Authorised Push Payment (APP) fraud investigation, benchmarked against an Isolation Forest baseline.

---

## 2. Problem Statement (final, reviewer-corrected version)

SEPA Instant Credit Transfers settle in under 10 seconds and are irreversible. Authorised Push Payment (APP) fraud — where victims are psychologically manipulated into willingly authorising payments to fraudster-controlled accounts — is the fastest-growing fraud type in this space, with EU losses of €2.5 billion in 2024 (up 24% YoY, per the EBA-ECB 2025 Payment Fraud Report). Because the payment is technically legitimate (correct authentication, genuine sender intent), the strongest available signal is **behavioural deviation** from an account's own normal payment history.

**Important framing correction (post-supervisor-feedback):** This project builds a **behavioural anomaly detector**, not a fraud classifier. Not every anomaly is fraud, and not every fraud type produces a strong statistical anomaly (see Results — invoice fraud). Flagged transactions are candidates for human investigation, not confirmed fraud verdicts.

---

## 3. Core Hypothesis

A decoder-only autoregressive Transformer, trained exclusively on normal per-account SEPA payment sequences, can surface behaviourally anomalous transactions — measured as low sequence likelihood — as effective candidates for APP fraud investigation, within the latency constraints of the SEPA Instant settlement window.

---

## 4. Final Scope (after supervisor feedback and 7-day timeline correction)

### In scope
1. **SynSEPA dataset** — synthetic SEPA Instant transaction dataset
2. **Decoder-only autoregressive Transformer** — the core detection model
3. **Isolation Forest baseline** — for comparison (kept in scope since already built/tested at no extra cost)
4. **Evaluation** — AUROC, PR-AUC, Precision/Recall/F1 @ calibrated threshold, plus a per-typology recall breakdown
5. **Light interpretability pass** — attention visualisation (framed explicitly as an interpretability aid, not formal explainability)

### Explicitly out of scope (documented as Future Work, not silently dropped)
- LSTM Autoencoder baseline
- CVAE / Diffusion-based synthetic fraud augmentation
- SHAP-based explainability
- Sequence-length ablation study
- Kaggle / HuggingFace public dataset publication
- Multi-typology fraud classification (the model detects anomalies, not typologies)
- LoRA/PEFT/QLoRA-based approaches (not applicable — no large pretrained transaction model exists to adapt; model is trained from scratch and is already smaller than a typical LoRA adapter)

---

## 5. Chronological Narrative of the Project (for report's Methodology / Development Process section)

1. Explored several project directions (fraud detection generally, an LLM+graph evidence-linking system called "EvidenceGraph") before converging on generative-model-based payments fraud detection.
2. Narrowed to SEPA Instant payments specifically after researching that no academic work or public dataset exists in this exact space, despite it being a large and growing real-world problem (EBA-ECB, EPC 2025 reports).
3. Learned and internalised foundational concepts needed for the project: the generative-modelling angle for anomaly detection, SEPA payment message structure, APP fraud typologies (per EPC 2025), the WeChat Pay GPT-style fraud detection paper, and the "Attention Is All You Need" Transformer paper (Q/K/V, self-attention vs cross-attention, multi-head attention, positional encoding, causal masking, residual connections, LayerNorm, feed-forward networks).
4. Investigated and **deliberately rejected** several candidate additions after research: purchase scams (weak SEPA-specific grounding, indistinguishable from normal peer transfers), the Santander `gen-fraud-graph` open-source tool (solves a different problem — cyclic money-laundering graphs — not sequential APP fraud), and PEFT/LoRA/QLoRA (not applicable to a from-scratch small model).
5. Designed and generated the SynSEPA dataset via a persona-based synthetic generation pipeline (Phase 2), calibrated against EBA-ECB 2025 statistics.
6. Tokenised the dataset and built per-account sliding-window sequences (Phase 3), deliberately avoiding raw high-cardinality field tokenisation (e.g. IBANs) to prevent memorisation instead of behavioural learning.
7. Implemented a decoder-only autoregressive Transformer from scratch in PyTorch (Phase 4) rather than using `nn.TransformerEncoderLayer`, to retain full control over causal masking and attention-weight extraction for later interpretability.
8. Received structured supervisor/reviewer feedback covering conceptual precision (anomaly vs fraud, "GPT-style" terminology, attention vs explanation), data/modelling rigour (per-account sequencing, IBAN tokenisation risk, threshold calibration), and evaluation completeness (missing baselines, missing metrics). Assessed each point honestly: most were documentation/terminology corrections rather than code defects, since per-account sequencing, non-LLM rule-based fraud generation, and non-raw-IBAN tokenisation were already implemented correctly but not clearly stated.
9. Revised the project reference document (v2) to correct terminology throughout, add an explicit threshold calibration protocol and evaluation protocol, and formally tighten scope to 4 core components given a realistic 7-day solo timeline.
10. Trained the Transformer in Google Colab (T4 GPU) and evaluated against the Isolation Forest baseline.
11. Diagnosed the headline recall figure (63%) by breaking it down per fraud typology, revealing a clear and explainable pattern (see Results).
12. Built attention visualisation for two illustrative examples (impersonation — caught; invoice — missed) as an interpretability aid, with explicit epistemic caveats.
13. Wrote an inference-time demo script showing how the trained model would be used in a real-time SEPA payment screening pipeline.

---

## 6. Dataset — SynSEPA

**What it is:** A fully synthetic SEPA Instant Credit Transfer transaction dataset, generated because no public SEPA fraud dataset exists.

**Scale:**
- 10,000 synthetic accounts across 4 personas (employee 50.5%, student 19.7%, retiree 15.2%, business 14.5%)
- 1,832,448 normal transactions (12 months simulated, Jan–Dec 2024)
- 7,112 fraud transactions across 4 APP typologies, rule-based (not LLM-generated), grounded in EPC 2025 documented behavioural signatures:
  - Bank/authority impersonation: 1,440 transactions
  - Invoice/mandate fraud: 900 transactions
  - Romance scam: 4,232 transactions (720 victims, ~5.9 payments/victim — multi-event by design)
  - CEO/BEC fraud: 540 transactions
- Overall fraud rate: 0.387% (vs EBA-ECB ~0.2% volume benchmark — the difference is explained by romance scam's multi-transaction nature)

**Validation:** 41 of 42 automated statistical checks passed (basic composition, fraud typology signatures, persona behavioural profiles, temporal patterns, cross-border rates, sequence integrity, amount distributions). The one failing check: retiree cross-border rate at 10.3% vs a 10.0% design target — a negligible calibration artefact, noted explicitly as a limitation.

**Important design decisions to carry into the report:**
- Fraud generation is **rule-based**, encoding documented EPC typology signatures — **not** LLM-generated
- Sequences are constructed **strictly per account** — never mixed across customers
- Raw IBAN strings are **never tokenised directly** — only derived low-cardinality features (`is_new_beneficiary`, `country_type`) are used, specifically to prevent the model from memorising specific account numbers instead of learning general behavioural patterns

**Not pursued/publication descoped:** Kaggle/HuggingFace publication package (README, datasheet, license) was fully prepared but publication itself is out of scope for the 7-day submission — noted as Future Work.

---

## 7. Model Architecture — SEPAGen Transformer

**Type:** Decoder-only autoregressive Transformer (explicitly *not* called "GPT-style" — that term is reserved for OpenAI's specific model family; the architecture uses the same underlying mechanism without being a GPT model).

**Hyperparameters:**
| Parameter | Value | Rationale |
|---|---|---|
| `d_model` | 64 | Embedding dimension per transaction; small vocabulary (66 tokens) doesn't need larger |
| `n_heads` | 4 | Must divide `d_model` evenly (64/4=16 per head); 4 parallel attention perspectives |
| `n_layers` | 2 | Shallow — tabular sequences are simpler than language; avoids overfitting on this data scale |
| `d_ff` | 256 | Standard 4× `d_model` expansion ratio (matches BERT/GPT convention) |
| `seq_len` | 20 | Transactions of context per sequence; covers ~1-4 months depending on persona |
| `vocab_size` | 66 | 65 combined token values across 8 fields (offset, non-overlapping) + 1 PAD token |
| `dropout` | 0.1 | Conservative regularisation given ~1.17M training sequences |
| Total parameters | 103,808 | Verified via architecture test |

**Tokenisation scheme (8 tokens per transaction, offset into one shared vocabulary):**
| Field | Values | Offset range |
|---|---|---|
| amount (bucketed) | 10 buckets | 0–9 |
| country_type | 3 (domestic/eu/non-eu) | 10–12 |
| remittance_category | 9 categories | 13–21 |
| is_new_beneficiary | 2 (binary) | 22–23 |
| is_weekend | 2 (binary) | 24–25 |
| hour_of_day | 24 | 26–49 |
| day_of_week | 7 | 50–56 |
| time_since_last_txn (bucketed) | 8 buckets | 57–64 |
| PAD | 1 | 65 |

**Architecture components (all implemented from scratch in PyTorch, not using `nn.TransformerEncoderLayer`):**
- `TransactionEmbedding` — sums 8 field-embeddings into one `d_model`-sized vector per transaction
- `PositionalEncoding` — standard sinusoidal encoding (Vaswani et al. 2017)
- `CausalSelfAttention` — scaled dot-product attention with upper-triangular causal mask (blocks attending to future transactions)
- `FeedForward` — two-layer MLP with GELU activation, 4× expansion
- `CausalTransformerBlock` — Pre-LN residual structure: `x = x + Attn(LayerNorm(x))`, `x = x + FF(LayerNorm(x))`
- `SEPAGenModel` — full stack; output projection weights are tied to the input embedding weights
- `AnomalyScorer` — computes mean log-likelihood across the 8 target tokens, given the preceding 20-transaction context

**Training setup:**
- AdamW optimiser, linear warmup + cosine decay schedule
- Batch size 256, 10 epochs
- Training data: **normal transactions only**, strictly time-split (train: Jan–Sep 2024; validation: Oct 2024; test: Nov–Dec 2024, contains fraud)
- Final result: train_loss=3.3287, val_loss=3.3317 (close — no overfitting, but suggests capacity/epoch ceiling not yet reached; noted as a limitation/future work item)

---

## 8. Threshold Calibration Protocol (added after reviewer feedback — previously undefined)

1. Score every validation-set transaction using the trained model's log-likelihood
2. Build the ROC curve from validation scores against `is_fraud` validation labels
3. Select the threshold at **1% false positive rate** — chosen as an operationally realistic rate for a bank's analyst review queue
4. Report additional operating points (0.5%, 2%, 5% FPR) for robustness (business-decision-dependent)
5. Apply the **frozen, validation-derived threshold** to the held-out test set for final reported metrics

---

## 9. Results

### 9.1 Headline comparison — SEPAGen Transformer vs Isolation Forest baseline

| Metric | Isolation Forest | SEPAGen Transformer | Change |
|---|---|---|---|
| AUROC | 0.9514 | 0.8927 | −0.0587 |
| PR-AUC | 0.1368 | 0.5981 | **+0.4613 (+337% relative)** |
| Precision @ 1% FPR | — | 0.3774 | — |
| Recall @ 1% FPR | — | 0.6309 | — |
| F1 @ 1% FPR | — | 0.4723 | — |
| Threshold @ 1% FPR | −0.0143 | 4.0268 | (different scoring scales) |

**Interpretation for the report:** AUROC is known to be an unreliable/inflated indicator under severe class imbalance (~0.4% fraud rate here). PR-AUC is the more appropriate metric for this setting, and the 337% relative improvement is the project's central empirical result: sequential, per-account behavioural modelling captures fraud signal that a per-transaction method (Isolation Forest) structurally cannot see.

### 9.2 Recall broken down by fraud typology (the most informative diagnostic result)

| Typology | Total (test) | Caught | Missed | Recall |
|---|---|---|---|---|
| Impersonation | 1,440 | 1,439 | 1 | **99.93%** |
| Invoice/mandate | 900 | 21 | 879 | **2.33%** |
| Romance scam | 54 | 37 | 17 | 68.52% |
| CEO/BEC | 540 | 354 | 186 | 65.56% |

**This fully explains the 63.09% blended recall** (weighted average of the above ≈ 0.631). This is a genuine, defensible finding rather than a weakness: impersonation, romance, and CEO fraud all produce strong behavioural deviations (unusual amount, new beneficiary, unusual timing/geography) that the model detects well. Invoice fraud is *designed*, per the EPC typology, to closely mimic the victim's normal payment amount and category — it differs primarily in the beneficiary IBAN, a feature this project deliberately does not tokenise directly (to avoid memorisation). This is consistent with real-world practice, where invoice/mandate fraud is primarily addressed via Verification of Payee (IBAN–name matching) rather than behavioural anomaly detection — supporting a "layered defence" framing in the report's discussion rather than treating this as a model failure.

### 9.3 Attention visualisation (interpretability, not explainability)

Two illustrative examples were visualised:
- An impersonation fraud case (correctly flagged) — expected to show concentrated attention on the transaction(s) establishing the account's normal amount baseline and/or the unusually short time gap before the fraudulent transaction
- An invoice fraud case (missed) — expected to show diffuse, undifferentiated attention, consistent with the absence of any standout anomalous prior transaction

**Explicit epistemic framing (per reviewer feedback and Jain & Wallace 2019, *"Attention is not Explanation"*):** attention weights indicate where the model's computation focused, not a verified causal reason for its score. Presented in the report as an interpretability aid for analyst triage, not formal model explainability.

---

## 10. Key Conceptual Corrections Made After Supervisor Feedback

| Original phrasing | Corrected phrasing | Reason |
|---|---|---|
| "Fraud detection" | "Behavioural anomaly detection" | Not every anomaly is fraud; not every fraud is anomalous (see invoice fraud result) |
| "GPT-style Transformer" | "Decoder-only autoregressive Transformer" | "GPT" refers specifically to OpenAI's model family |
| "Attention-based explainability" | "Attention-based interpretability signal" | Attention shows focus, not verified causation (Jain & Wallace, 2019) |
| (implicit) sequences per account | Explicitly stated: sequences are constructed strictly per account, never mixed | Was already implemented correctly, just not stated clearly in the original proposal |
| (implicit) fraud generation method | Explicitly stated: rule-based, not LLM-based, grounded in EPC 2025 documented signatures | Same — correct implementation, unclear original documentation |
| (implicit) tokenisation of IBANs | Explicitly stated: raw IBANs are never tokenised; only derived low-cardinality features are used | Same — correct implementation, unclear original documentation; also now framed as a deliberate strength |
| (undefined) | Explicit threshold calibration protocol added | Was genuinely missing before |
| (incomplete) | Explicit evaluation protocol with full metric rationale added | Was genuinely missing before |

---

## 11. Complete File Inventory

### 11.1 Data generation pipeline (Phase 2)

| File | Purpose | Key output |
|---|---|---|
| `step1_generate_accounts.py` | Generates 10,000 synthetic accounts with persona assignment (employee/student/retiree/business), IBANs, known-beneficiary lists, behavioural parameters | `accounts.csv` |
| `step2_generate_normal_transactions.py` | Generates 12 months of normal transaction history per account, using persona-specific amount/timing/category distributions | `normal_transactions.csv` (~1.83M rows) |
| `step3_inject_fraud.py` | Injects 4 APP fraud typologies (impersonation, invoice, romance, CEO) as rule-based behavioural signatures per EPC 2025, targeting eligible personas | `synsep_full_dataset.csv` (final combined dataset, ~1.84M rows) |
| `step4_validate_dataset.py` | Runs 42 automated statistical validation checks against EBA-ECB benchmarks and internal consistency rules | `validation_report.txt` (41/42 passed) |

### 11.2 Data preparation pipeline (Phase 3)

| File | Purpose | Key output |
|---|---|---|
| `phase3_step1_tokenise.py` | Converts each of the 8 transaction fields into offset integer tokens using a single 66-token shared vocabulary; explicitly avoids tokenising raw IBANs | `tokenised_dataset.csv`, `tokeniser.json` (vocabulary/bucket definitions — required for any future inference) |
| `phase3_step2_sequences.py` | Builds per-account sliding-window sequences (20 transactions of context → predict the 21st); performs the time-based train/val/test split (train: Jan–Sep normal-only; val: Oct; test: Nov–Dec) | `sequences_train.csv`, `sequences_val.csv`, `sequences_test.csv` |

### 11.3 Model, training, and evaluation (Phase 4)

| File | Purpose | Key output |
|---|---|---|
| `phase4_transformer.py` | Full decoder-only autoregressive Transformer implementation (embedding, positional encoding, causal self-attention, feed-forward, training loop, AnomalyScorer, evaluation). Supports `--test` flag for a fast architecture sanity check without real data. | `sepagen_model.pt` (trained checkpoint), `training_log.csv`, printed AUROC/PR-AUC/F1 results |
| `sample_baseline.py` | Isolation Forest baseline using identical time-based split and feature engineering, for fair comparison against the Transformer | Printed AUROC 0.9514 / PR-AUC 0.1368 results |
| `diagnose_recall_by_typology.py` | Loads the trained model and breaks the overall recall figure down per fraud typology, plus reports mean anomaly score per typology vs the calibrated threshold | `recall_by_typology.json`, printed per-typology recall table |
| `attention_visualisation.py` | Extracts and visualises last-layer attention weights (averaged across heads) for one impersonation example (caught) and one invoice example (missed), with token-decoding back to human-readable transaction summaries | `attention_impersonation.png`, `attention_invoice.png`, `attention_comparison.png` |
| `inference_demo.py` | Demonstrates end-to-end real-world inference: loading the frozen trained model, tokenising a new transaction using the saved `tokeniser.json`, scoring it against an account's recent history, and applying the calibrated threshold — with a description of the surrounding operational pipeline (analyst review queue, SLA, retraining loop) | Printed worked example (retiree account, suspicious €8,500 transfer) |

### 11.4 Documentation and reference

| File | Purpose |
|---|---|
| `SEPAGen_Project_Reference.md` | **v1** — original project plan (4-week timeline, 8 phases). Superseded by v2. |
| `SEPAGen_Project_Reference_v2.md` | **v2 — current** — revised after supervisor feedback: corrected terminology throughout, added threshold calibration protocol, added evaluation protocol, tightened scope to 7-day/4-component plan, explicit Future Work list |
| `SEPAGen_Architecture.svg` | Visual system architecture diagram (input → tokenisation → Transformer → scoring → threshold → output → explainability → baselines) |
| `README.md` | Dataset card for SynSEPA (Kaggle/HuggingFace-ready format) — statistics, schema, typology definitions, generation methodology, intended uses/limitations, citation |
| `datasheet.md` | Formal "Datasheets for Datasets" (Gebru et al. 2021) responsible-AI documentation for SynSEPA |
| `validation_report.txt` | Output of the 42 automated dataset validation checks (41 passed) |
| `accounts.csv` | 10,000-row account metadata table (persona, IBAN, home country, known beneficiaries, behavioural parameters) |
| `tokeniser.json` | Saved vocabulary/bucket-boundary definitions — required for reproducing tokenisation identically at inference time |

### 11.5 Large data files (present but likely excluded from git via .gitignore — see Section 13)

| File | Size | Notes |
|---|---|---|
| `synsep_full_dataset.csv` | ~299 MB | Final combined dataset — likely too large for a standard GitHub repo without Git LFS |
| `normal_transactions.csv` | ~297 MB | Intermediate output of Step 2 — likely excludable, regenerable from scripts |

---

## 12. Papers and Sources Referenced Throughout

| Source | Role in project |
|---|---|
| Vaswani et al. (2017), *Attention Is All You Need* | Core Transformer architecture — Q/K/V, multi-head attention, positional encoding |
| Zhao et al. (2023), *Generative Pretraining at Scale for Fraud Detection* (WeChat Pay/Tencent) | Closest prior work — same autoregressive-sequence framing, same anomaly-not-typology framing; token-explosion handling and differential-convolution ideas noted as related but not implemented |
| Jain & Wallace (2019), *Attention Is Not Explanation* | Justifies framing attention visualisation as interpretability, not explainability |
| EBA-ECB (2025), *Joint Report on Payment Fraud* | Real-world statistics used to calibrate SynSEPA (fraud rate ~0.2%, €2.5B credit transfer losses, cross-border rates, SEPA Instant fraud growth of 175%) |
| EPC (2025), *Payment Threats and Fraud Trends Report* (EPC162-24 v2.0) | Source of the documented APP fraud typology behavioural signatures used to construct the 4 rule-based fraud generators |
| Gebru et al. (2021), *Datasheets for Datasets* | Format followed for `datasheet.md` |
| Lopez-Rojas (2017), PaySim | Referenced as prior synthetic financial dataset methodology (persona/agent-based simulation approach) |

---

## 13. Suggested GitHub Repository Structure

This is a suggested organisation for Claude Code to implement — grouping by pipeline phase, separating code from generated data/docs, and flagging large files for `.gitignore` or Git LFS consideration.

```
sepagen/
├── README.md                          # Project-level README (adapt from SynSEPA README.md + add model results)
├── LICENSE                            # CC BY 4.0 (as stated in existing README.md)
├── .gitignore                         # Exclude large CSVs, .pt checkpoints, __pycache__ etc.
│
├── docs/
│   ├── project_reference_v1.md        # ← SEPAGen_Project_Reference.md (historical)
│   ├── project_reference_v2.md        # ← SEPAGen_Project_Reference_v2.md (current/authoritative)
│   ├── architecture.svg               # ← SEPAGen_Architecture.svg
│   ├── datasheet.md                   # ← datasheet.md
│   └── validation_report.txt          # ← validation_report.txt
│
├── src/
│   ├── data_generation/
│   │   ├── step1_generate_accounts.py
│   │   ├── step2_generate_normal_transactions.py
│   │   ├── step3_inject_fraud.py
│   │   └── step4_validate_dataset.py
│   │
│   ├── data_preparation/
│   │   ├── phase3_step1_tokenise.py
│   │   └── phase3_step2_sequences.py
│   │
│   ├── model/
│   │   └── phase4_transformer.py
│   │
│   └── evaluation/
│       ├── sample_baseline.py
│       ├── diagnose_recall_by_typology.py
│       ├── attention_visualisation.py
│       └── inference_demo.py
│
├── data/
│   ├── accounts.csv                   # small enough to commit directly (~4MB)
│   ├── tokeniser.json                 # small — commit directly
│   └── README.md                      # note: large CSVs (synsep_full_dataset.csv,
│                                       # normal_transactions.csv, sequences_*.csv,
│                                       # tokenised_dataset.csv) are NOT committed —
│                                       # regenerate via scripts in src/data_generation/
│                                       # and src/data_preparation/, or fetch from
│                                       # [Kaggle/HuggingFace link once published]
│
├── outputs/
│   ├── sepagen_model.pt                # trained checkpoint — consider Git LFS or
│                                        # excluding + documenting how to regenerate
│   ├── training_log.csv
│   ├── recall_by_typology.json
│   ├── attention_impersonation.png
│   ├── attention_invoice.png
│   └── attention_comparison.png
│
└── report/
    └── (final report generated from this summary — see Section 14)
```

**Suggested `.gitignore` contents:**
```
*.csv
!data/accounts.csv
*.pt
__pycache__/
*.pyc
.ipynb_checkpoints/
```

**Note for Claude Code:** the raw dataset CSVs are large (~300MB each) and were generated by deterministic scripts with a fixed random seed (42) — they are fully reproducible by re-running `src/data_generation/step1_generate_accounts.py` through `step4_validate_dataset.py` in sequence, followed by `src/data_preparation/phase3_step1_tokenise.py` and `phase3_step2_sequences.py`. Recommend documenting this regeneration path in `data/README.md` rather than committing the large files, unless Git LFS is set up.

---

## 14. Guidance for Final Report Generation

The final report should follow this structure (as already specified in `SEPAGen_Project_Reference_v2.md`, Section "Day 5–6 — Write-Up"):

1. **Abstract**
2. **Introduction** — problem, motivation, and the anomaly-vs-fraud framing (Section 2 above)
3. **Related Work** — WeChat Pay paper, Jain & Wallace on attention (Section 12 above)
4. **Dataset** — SynSEPA generation methodology, explicit rule-based (not LLM-based) fraud construction, validation results (Section 6 above)
5. **Methodology** — architecture, per-account sequencing rationale, no-raw-IBAN tokenisation rationale, threshold calibration protocol (Sections 7–8 above)
6. **Experiments** — evaluation protocol, Transformer vs Isolation Forest results (Section 9.1 above)
7. **Results Discussion** — per-typology recall breakdown and its interpretation (Section 9.2 above) — this is the most substantive analytical contribution and should be given real weight
8. **Interpretability** — attention visualisation with explicit epistemic caveats (Section 9.3 above)
9. **Limitations & Future Work** — explicitly list every descoped item from Section 4 ("Explicitly out of scope"), plus the retiree cross-border calibration note and the close train/val loss observation
10. **Conclusion**
11. **References** (Section 12 above)

All terminology throughout the report must follow the corrected forms in Section 10 — this was the single most substantive piece of supervisor feedback and consistency here matters more than any other stylistic choice.

---

*This document was compiled as a complete handoff summary of the SEPAGen project as of the current development state (Phases 1–4 complete and trained; Day 4 interpretability pass in progress; write-up not yet started). It is intended as the single source of truth for both repository organisation and final report drafting.*
