# Data — Generation Guide

The large CSV files are not committed to this repository (see `.gitignore`).
You have two options to get the data:

**Option A — Download pre-generated dataset (recommended)**
Download `synsep_full_dataset.csv` and `accounts.csv` from the SynSEPA dataset repository
and place them in this directory (`data/`) or the repo root.

**Option B — Regenerate from scratch (takes ~10–15 minutes)**

Run each step from the **repo root** in order:

---

### Step 1 — Generate 10,000 synthetic accounts

```bash
python src/data_generation/step1_generate_accounts.py
```

Output: `accounts.csv`

---

### Step 2 — Generate 12 months of normal transactions

```bash
python src/data_generation/step2_generate_normal_transactions.py
```

Output: `normal_transactions.csv` (~1.83M rows)

---

### Step 3 — Inject APP fraud transactions

```bash
python src/data_generation/step3_inject_fraud.py
```

Output: `synsep_full_dataset.csv` (~1.84M rows)

---

### Step 4 — Validate dataset against EBA-ECB benchmarks (optional)

```bash
python src/data_generation/step4_validate_dataset.py
```

Output: `validation_report.txt`

---

### Phase 3, Step 1 — Tokenise transactions

```bash
python src/preprocessing/phase3_step1_tokenise.py
```

Output: `tokenised_dataset.csv` , `tokeniser.json` (already committed)

---

### Phase 3, Step 2 — Build sliding-window sequences

```bash
python src/preprocessing/phase3_step2_sequences.py
```

Output: `sequences_train.csv`, `sequences_val.csv`, `sequences_test.csv`

---

The `tokeniser.json` file is committed to this repo — it defines the vocabulary
and bucket definitions and is required for model inference.
