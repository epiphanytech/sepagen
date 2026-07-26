"""
SEPAGen - Phase 3, Step 1: Tokenisation
=========================================
Converts each SEPA transaction field into an integer token.

Why tokenise?
  A Transformer works on sequences of discrete tokens — like words in a sentence.
  Each transaction becomes a sequence of tokens, one per field.
  The model learns to predict the next token, capturing normal payment patterns.

Tokenisation strategy per field:
  - Continuous fields (amount, time_delta) → bucketed into N discrete ranges
  - Categorical fields (country_type, category) → integer label encoding
  - Binary fields (is_new_beneficiary, is_weekend) → kept as 0/1
  - Time fields (hour, day_of_week) → kept as integers (already discrete)

Output:
  - tokeniser.json  ← vocabulary and bucket definitions (save for inference)
  - tokenised_dataset.csv ← full dataset with token columns added
"""

import csv
import json
import math
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

INPUT_FILE      = "synsep_full_dataset.csv"
OUTPUT_FILE     = "tokenised_dataset.csv"
TOKENISER_FILE  = "tokeniser.json"

# ─────────────────────────────────────────
# TOKENISATION DESIGN
# ─────────────────────────────────────────
#
# Each transaction becomes a fixed-length token sequence:
#
# Token 0: amount_token        (0-9,  10 buckets)
# Token 1: country_token       (0-2,  3 values: domestic/eu/non-eu)
# Token 2: category_token      (0-8,  9 categories)
# Token 3: is_new_ben_token    (0-1,  binary)
# Token 4: is_weekend_token    (0-1,  binary)
# Token 5: hour_token          (0-23, 24 values)
# Token 6: day_token           (0-6,  7 values)
# Token 7: time_delta_token    (0-7,  8 buckets)
#
# Total: 8 tokens per transaction
# Vocabulary sizes combined into single offset vocabulary for embedding
#
# ─────────────────────────────────────────

# Amount buckets (EUR) — 10 buckets
# Designed to cover student (€5) to business (€50k) range
AMOUNT_BUCKETS = [
    (0,      50),     # 0: micro      — small daily purchases
    (50,     150),    # 1: small-low  — groceries
    (150,    300),    # 2: small-high — utilities
    (300,    600),    # 3: medium-low — rent contribution
    (600,    1200),   # 4: medium     — rent
    (1200,   2500),   # 5: medium-high
    (2500,   5000),   # 6: large-low
    (5000,   10000),  # 7: large
    (10000,  25000),  # 8: very large
    (25000,  float("inf")),  # 9: extra large — business/fraud
]

# Time delta buckets (seconds since last transaction) — 8 buckets
# NULL (first transaction) → bucket 0
TIME_DELTA_BUCKETS = [
    (0,        3600),        # 0: < 1 hour      → suspicious velocity
    (3600,     21600),       # 1: 1-6 hours
    (21600,    86400),       # 2: 6-24 hours
    (86400,    259200),      # 3: 1-3 days
    (259200,   604800),      # 4: 3-7 days
    (604800,   1209600),     # 5: 1-2 weeks
    (1209600,  2592000),     # 6: 2-4 weeks
    (2592000,  float("inf")),# 7: > 1 month
]

# Country type mapping
COUNTRY_TYPE_MAP = {
    "domestic":         0,
    "eu_cross_border":  1,
    "non_eu":           2,
}

# Remittance category mapping
CATEGORY_MAP = {
    "grocery":      0,
    "rent":         1,
    "utility":      2,
    "shopping":     3,
    "transfer":     4,
    "subscription": 5,
    "supplier":     6,
    "salary":       7,
    "other":        8,
}

# ─────────────────────────────────────────
# VOCABULARY OFFSETS
# We use a single embedding table with offsets
# so each field's tokens don't overlap
# ─────────────────────────────────────────
#
# Field         Size    Offset  Range
# amount        10      0       0-9
# country       3       10      10-12
# category      9       13      13-21
# is_new_ben    2       22      22-23
# is_weekend    2       24      24-25
# hour          24      26      26-49
# day           7       50      50-56
# time_delta    8       57      57-64
#
# Total vocabulary size: 65 tokens
# ─────────────────────────────────────────

OFFSETS = {
    "amount":       0,
    "country":      10,
    "category":     13,
    "is_new_ben":   22,
    "is_weekend":   24,
    "hour":         26,
    "day":          50,
    "time_delta":   57,
}

VOCAB_SIZE = 65

# Special tokens
PAD_TOKEN = VOCAB_SIZE      # padding token (for sequences shorter than max length)
VOCAB_SIZE_WITH_SPECIAL = VOCAB_SIZE + 1   # 66 total including PAD

# ─────────────────────────────────────────
# TOKENISATION FUNCTIONS
# ─────────────────────────────────────────

def bucket(value, buckets):
    """
    Find which bucket a value falls into.
    Returns bucket index (0-based).
    """
    for i, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return len(buckets) - 1   # overflow → last bucket


def tokenise_transaction(row):
    """
    Convert a single transaction row into a list of 8 integer tokens.
    Each token is offset so it falls in the correct vocabulary range.

    Returns: list of 8 integers (the token sequence for this transaction)
    """

    # ── Token 0: Amount ──
    amount      = float(row.get("amount", 0))
    amount_tok  = bucket(amount, AMOUNT_BUCKETS) + OFFSETS["amount"]

    # ── Token 1: Country type ──
    country     = row.get("country_type", "domestic")
    country_tok = COUNTRY_TYPE_MAP.get(country, 0) + OFFSETS["country"]

    # ── Token 2: Remittance category ──
    category     = row.get("remittance_category", "other")
    category_tok = CATEGORY_MAP.get(category, 8) + OFFSETS["category"]

    # ── Token 3: Is new beneficiary ──
    is_new      = int(row.get("is_new_beneficiary", 0))
    is_new_tok  = is_new + OFFSETS["is_new_ben"]

    # ── Token 4: Is weekend ──
    is_wknd     = int(row.get("is_weekend", 0))
    is_wknd_tok = is_wknd + OFFSETS["is_weekend"]

    # ── Token 5: Hour of day ──
    hour        = int(float(row.get("hour_of_day", 12)))
    hour_tok    = hour + OFFSETS["hour"]

    # ── Token 6: Day of week ──
    day         = int(float(row.get("day_of_week", 0)))
    day_tok     = day + OFFSETS["day"]

    # ── Token 7: Time since last transaction ──
    delta_raw   = row.get("time_since_last_txn", None)
    if delta_raw in (None, "", "None"):
        # First transaction — no previous — use bucket 7 (long gap)
        delta_tok = 7 + OFFSETS["time_delta"]
    else:
        delta_sec = float(delta_raw)
        delta_tok = bucket(delta_sec, TIME_DELTA_BUCKETS) + OFFSETS["time_delta"]

    return [
        amount_tok,
        country_tok,
        category_tok,
        is_new_tok,
        is_wknd_tok,
        hour_tok,
        day_tok,
        delta_tok,
    ]


# ─────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────

def process_dataset(input_file, output_file):
    """
    Read full dataset, tokenise every transaction,
    write tokenised dataset to CSV.
    """
    print(f"Tokenising {input_file}...")
    print("(Processing 1.84M rows — ~30 seconds)\n")

    # Track stats for validation
    stats = defaultdict(lambda: defaultdict(int))
    total = 0

    with open(input_file,  "r", encoding="utf-8") as fin, \
         open(output_file, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)

        # Add token columns to output
        fieldnames = list(reader.fieldnames) + [
            "tok_amount", "tok_country", "tok_category",
            "tok_is_new_ben", "tok_is_weekend",
            "tok_hour", "tok_day", "tok_time_delta",
            "token_sequence",   # full sequence as JSON string
        ]

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            tokens = tokenise_transaction(row)

            # Add individual token columns
            row["tok_amount"]      = tokens[0]
            row["tok_country"]     = tokens[1]
            row["tok_category"]    = tokens[2]
            row["tok_is_new_ben"]  = tokens[3]
            row["tok_is_weekend"]  = tokens[4]
            row["tok_hour"]        = tokens[5]
            row["tok_day"]         = tokens[6]
            row["tok_time_delta"]  = tokens[7]
            row["token_sequence"]  = json.dumps(tokens)

            writer.writerow(row)

            # Track distribution for validation
            for i, tok in enumerate(tokens):
                stats[i][tok] += 1

            total += 1
            if total % 500_000 == 0:
                print(f"  → {total:,} transactions tokenised...")

    print(f"\n✅ Tokenised {total:,} transactions → {output_file}")
    return stats


def save_tokeniser(output_file):
    """
    Save tokeniser vocabulary and bucket definitions.
    IMPORTANT: This must be saved and reloaded at inference time
    so new transactions are tokenised identically to training data.
    """
    tokeniser = {
        "vocab_size":             VOCAB_SIZE,
        "vocab_size_with_special": VOCAB_SIZE_WITH_SPECIAL,
        "pad_token":              PAD_TOKEN,
        "tokens_per_transaction": 8,
        "offsets":                OFFSETS,
        "amount_buckets":         [list(b) if b[1] != float("inf")
                                   else [b[0], None]
                                   for b in AMOUNT_BUCKETS],
        "time_delta_buckets":     [list(b) if b[1] != float("inf")
                                   else [b[0], None]
                                   for b in TIME_DELTA_BUCKETS],
        "country_type_map":       COUNTRY_TYPE_MAP,
        "category_map":           CATEGORY_MAP,
        "field_order": [
            "amount", "country", "category",
            "is_new_ben", "is_weekend",
            "hour", "day", "time_delta"
        ],
        "field_descriptions": {
            "amount":       "EUR amount bucket (0=<€50, 9=>€25k)",
            "country":      "0=domestic, 1=EU cross-border, 2=non-EU",
            "category":     "Remittance category (0=grocery...8=other)",
            "is_new_ben":   "0=known beneficiary, 1=new beneficiary",
            "is_weekend":   "0=weekday, 1=weekend",
            "hour":         "Hour of day (0-23)",
            "day":          "Day of week (0=Mon, 6=Sun)",
            "time_delta":   "Time since last txn (0=<1hr, 7=>1month)",
        }
    }

    with open(output_file, "w") as f:
        json.dump(tokeniser, f, indent=2)

    print(f"✅ Tokeniser saved → {output_file}")
    return tokeniser


def print_token_stats(stats):
    """Print distribution of tokens per field."""
    field_names = [
        "amount", "country", "category",
        "is_new_ben", "is_weekend",
        "hour", "day", "time_delta"
    ]
    offsets_list = [
        OFFSETS["amount"],   OFFSETS["country"],
        OFFSETS["category"], OFFSETS["is_new_ben"],
        OFFSETS["is_weekend"], OFFSETS["hour"],
        OFFSETS["day"],      OFFSETS["time_delta"],
    ]

    print("\n" + "="*60)
    print("TOKEN DISTRIBUTION PER FIELD")
    print("="*60)

    for i, (name, offset) in enumerate(zip(field_names, offsets_list)):
        field_stats = stats[i]
        total = sum(field_stats.values())
        print(f"\n{name.upper()} (offset={offset}):")
        for tok in sorted(field_stats.keys()):
            local_val = tok - offset
            pct       = field_stats[tok] / total * 100
            bar       = "█" * int(pct / 2)
            print(f"  [{local_val:2d}] {bar:<50} {pct:5.1f}%")


def validate_tokens(stats):
    """Quick sanity check — all tokens in valid vocabulary range."""
    print("\n" + "="*60)
    print("TOKEN RANGE VALIDATION")
    print("="*60)

    all_valid = True
    for field_idx, field_stats in stats.items():
        for tok in field_stats.keys():
            if tok < 0 or tok >= VOCAB_SIZE:
                print(f"❌ INVALID token {tok} in field {field_idx}")
                all_valid = False

    if all_valid:
        print(f"✅ All tokens within valid range [0, {VOCAB_SIZE-1}]")
        print(f"   Vocabulary size: {VOCAB_SIZE} + 1 PAD = {VOCAB_SIZE_WITH_SPECIAL}")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*60)
    print("SEPAGen — Phase 3, Step 1: Tokenisation")
    print("="*60)

    print(f"\nVocabulary design:")
    print(f"  Tokens per transaction: 8")
    print(f"  Vocabulary size:        {VOCAB_SIZE}")
    print(f"  PAD token:              {PAD_TOKEN}")
    print(f"  Total vocab with PAD:   {VOCAB_SIZE_WITH_SPECIAL}")

    # Save tokeniser definition
    save_tokeniser(TOKENISER_FILE)

    # Process full dataset
    stats = process_dataset(INPUT_FILE, OUTPUT_FILE)

    # Print distribution stats
    print_token_stats(stats)

    # Validate token ranges
    validate_tokens(stats)

    print(f"\n✅ Phase 3, Step 1 complete")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Tokeniser: {TOKENISER_FILE}")
