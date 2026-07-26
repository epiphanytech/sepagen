"""
SEPAGen - Phase 3, Step 2: Sequence Creation
Lightweight version — writes CSV instead of numpy
to avoid memory issues in constrained environments.
Run this in Google Colab for numpy version.
"""

import csv
import json
from collections import defaultdict

INPUT_FILE     = "tokenised_dataset.csv"
TOKENISER_FILE = "tokeniser.json"
SEQ_LEN        = 20
TOKENS_PER_TXN = 8
TRAIN_END      = "2024-10-01"
VAL_END        = "2024-11-01"

def main():
    print("SEPAGen — Phase 3, Step 2: Sequence Creation")
    print("=" * 55)

    with open(TOKENISER_FILE) as f:
        tokeniser = json.load(f)
    pad_token = tokeniser["pad_token"]

    # Load transactions grouped by account
    print(f"\nLoading transactions...")
    account_txns = defaultdict(list)
    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            account_txns[row["account_id"]].append(row)
            total += 1
            if total % 500_000 == 0:
                print(f"  → {total:,} rows loaded...")

    for acc_id in account_txns:
        account_txns[acc_id].sort(key=lambda r: r["timestamp"])
    print(f"✅ {total:,} transactions across {len(account_txns):,} accounts")

    # Open output CSVs — one per split
    # Each row = one sequence stored as JSON strings
    split_files  = {}
    split_writers = {}
    split_counts  = {"train": 0, "val": 0, "test": 0}
    split_fraud   = {"train": 0, "val": 0, "test": 0}

    fieldnames = ["input_tokens", "target_tokens",
                  "account_id", "persona", "timestamp",
                  "is_fraud", "fraud_type", "amount"]

    for sname in ["train", "val", "test"]:
        f = open(f"sequences_{sname}.csv", "w",
                 newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        split_files[sname]   = f
        split_writers[sname] = w

    print(f"\nBuilding sequences (SEQ_LEN={SEQ_LEN})...")
    processed  = 0
    n_accounts = len(account_txns)

    for acc_id, txns in account_txns.items():

        token_seqs = [json.loads(t["token_sequence"]) for t in txns]
        n_txns     = len(token_seqs)

        if n_txns < 2:
            processed += 1
            continue

        if n_txns <= SEQ_LEN:
            pad_needed = SEQ_LEN - n_txns + 1
            token_seqs = [[pad_token]*TOKENS_PER_TXN]*pad_needed + token_seqs
            pad_offset = pad_needed
        else:
            pad_offset = 0

        for i in range(len(token_seqs) - SEQ_LEN):
            inp_win  = token_seqs[i : i + SEQ_LEN]
            tgt_toks = token_seqs[i + SEQ_LEN]

            orig_idx = i + SEQ_LEN - pad_offset
            tgt_txn  = txns[orig_idx] if 0 <= orig_idx < n_txns else txns[-1]

            ts       = tgt_txn.get("timestamp", "")
            is_fraud = int(tgt_txn.get("is_fraud", 0))

            # Assign split
            if ts < TRAIN_END:
                split = "train" if is_fraud == 0 else None
            elif ts < VAL_END:
                split = "val"
            else:
                split = "test"

            if split is None:
                continue

            split_writers[split].writerow({
                "input_tokens":  json.dumps(inp_win),
                "target_tokens": json.dumps(tgt_toks),
                "account_id":    acc_id,
                "persona":       tgt_txn.get("persona", ""),
                "timestamp":     ts,
                "is_fraud":      is_fraud,
                "fraud_type":    tgt_txn.get("fraud_type", "none"),
                "amount":        tgt_txn.get("amount", 0),
            })
            split_counts[split] += 1
            split_fraud[split]  += is_fraud

        processed += 1
        if processed % 1000 == 0:
            total_so_far = sum(split_counts.values())
            print(f"  → {processed:,}/{n_accounts:,} accounts | "
                  f"{total_so_far:,} sequences")

    for f in split_files.values():
        f.close()

    print("\n" + "=" * 55)
    print("SEQUENCE DATASET SUMMARY")
    print("=" * 55)
    print(f"SEQ_LEN={SEQ_LEN}  tokens_per_txn={TOKENS_PER_TXN}")
    print()
    for sname in ["train", "val", "test"]:
        print(f"  {sname.upper():<6} {split_counts[sname]:>10,} sequences  "
              f"fraud={split_fraud[sname]:,}")
    print()
    print("✅ Phase 3, Step 2 complete")
    print()
    print("NOTE: Sequences saved as CSV for portability.")
    print("In Google Colab, convert to numpy with:")
    print("  import numpy as np, json, csv")
    print("  rows = list(csv.DictReader(open('sequences_train.csv')))")
    print("  X = np.array([json.loads(r['input_tokens']) for r in rows], dtype=np.int32)")
    print("  y = np.array([json.loads(r['target_tokens']) for r in rows], dtype=np.int32)")
    print("=" * 55)

if __name__ == "__main__":
    main()
