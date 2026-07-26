"""
SEPAGen - Inference Demo
Shows how the trained model scores a new incoming transaction.

Usage (run from repo root):
    python src/model/inference_demo.py
"""

import json
import torch
from src.model.phase4_transformer import SEPAGenModel, AnomalyScorer, DEVICE, MODEL_SAVE

# Load trained model
checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
config = checkpoint["config"]

model = SEPAGenModel(**{k: config[k] for k in
    ["vocab_size", "d_model", "n_heads", "n_layers", "d_ff", "seq_len", "pad_token", "dropout"]
}).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model loaded  (val_loss={checkpoint['val_loss']:.4f})\n")

# Load tokeniser
with open("data/tokeniser.json") as f:
    tok = json.load(f)

AMOUNT_BUCKETS     = [(b[0], b[1] if b[1] is not None else float("inf")) for b in tok["amount_buckets"]]
TIME_DELTA_BUCKETS = [(b[0], b[1] if b[1] is not None else float("inf")) for b in tok["time_delta_buckets"]]
OFFSETS      = tok["offsets"]
COUNTRY_MAP  = tok["country_type_map"]
CATEGORY_MAP = tok["category_map"]


def bucket(value, buckets):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1


def tokenise(txn):
    delta = txn["time_since_last_txn"]
    return [
        bucket(txn["amount"], AMOUNT_BUCKETS)              + OFFSETS["amount"],
        COUNTRY_MAP.get(txn["country_type"], 0)            + OFFSETS["country"],
        CATEGORY_MAP.get(txn["remittance_category"], 8)    + OFFSETS["category"],
        int(txn["is_new_beneficiary"])                     + OFFSETS["is_new_ben"],
        int(txn["is_weekend"])                             + OFFSETS["is_weekend"],
        int(txn["hour_of_day"])                            + OFFSETS["hour"],
        int(txn["day_of_week"])                            + OFFSETS["day"],
        (7 if delta is None else bucket(delta, TIME_DELTA_BUCKETS)) + OFFSETS["time_delta"],
    ]


# Example: retiree account with 20 normal transactions
history = [
    {
        "amount": 45.0 + (i % 5) * 10,
        "country_type": "domestic",
        "remittance_category": "grocery" if i % 2 == 0 else "utility",
        "is_new_beneficiary": 0,
        "is_weekend": 0,
        "hour_of_day": 10,
        "day_of_week": i % 5,
        "time_since_last_txn": 259200,   # ~3 days
    }
    for i in range(20)
]

# Incoming transaction to score
new_txn = {
    "amount": 8500.0,             # 100x their normal amount
    "country_type": "eu_cross_border",
    "remittance_category": "transfer",
    "is_new_beneficiary": 1,
    "is_weekend": 0,
    "hour_of_day": 23,            # 11pm — unusual for this account
    "day_of_week": 2,
    "time_since_last_txn": 1800,  # 30 min after previous txn
}

# Score
input_tensor  = torch.tensor([[tokenise(t) for t in history]], dtype=torch.long).to(DEVICE)
target_tensor = torch.tensor([tokenise(new_txn)],              dtype=torch.long).to(DEVICE)

scorer = AnomalyScorer(model, DEVICE)
score  = scorer.score_batch(input_tensor, target_tensor)[0]

# Threshold from evaluation at 1% FPR — update this from your actual eval run
THRESHOLD = -4.0268

print(f"Account history:   €45–95 grocery/utility, domestic, weekdays 10am")
print(f"Incoming txn:      €{new_txn['amount']:.0f}, cross-border, new IBAN, 11pm, 30min gap")
print(f"\nAnomaly score:     {score:.4f}  (threshold: {THRESHOLD:.4f})")
print(f"Decision:          {'FLAGGED' if score < THRESHOLD else 'normal'}")
