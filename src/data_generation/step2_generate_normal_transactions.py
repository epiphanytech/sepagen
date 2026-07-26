"""
SEPAGen - Step 2: Normal Transaction Generation
=================================================
Generates 12 months of realistic normal SEPA transaction
history for all 10,000 accounts from Step 1.

Each transaction has:
  - Core SEPA fields (IBAN, amount, timestamp, reference)
  - Engineered features (time_since_last_txn, is_new_beneficiary, etc.)
  - Fraud label = 0 (all normal in this step)

Output: normal_transactions.csv
"""

import csv
import json
import random
from datetime import datetime, timedelta

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

RANDOM_SEED      = 42
ACCOUNTS_FILE    = "accounts.csv"
OUTPUT_FILE      = "normal_transactions.csv"

# Simulation period — 12 months
SIM_START = datetime(2024, 1, 1, 0, 0, 0)
SIM_END   = datetime(2024, 12, 31, 23, 59, 59)
SIM_DAYS  = (SIM_END - SIM_START).days

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────
# PERSONA DEFINITIONS (must match Step 1)
# Keeping only what Step 2 needs
# ─────────────────────────────────────────

PERSONAS = {
    "employee": {
        "txn_per_month_min": 8,
        "txn_per_month_max": 15,
        "amount_ranges": [
            (0.30, 10,   100),
            (0.35, 100,  500),
            (0.25, 500,  1500),
            (0.10, 1500, 3000),
        ],
        # active_hours: 8,9,10,11,12,13,14,15,16,17,18,19,20 = 13 hours
        "active_hours": list(range(8, 21)),
        "hour_weights": [5,5,8,10,10,10,10,10,8,8,5,5,3],  # 13 weights
        "weekend_factor": 0.6,
        "foreign_txn_probability": 0.05,
        "foreign_country_pool": ["FR","ES","IT","NL","PT"],
        "remittance_categories": {
            "rent": 0.10, "grocery": 0.25, "utility": 0.10,
            "shopping": 0.20, "transfer": 0.15,
            "subscription": 0.10, "other": 0.10,
        },
    },
    "student": {
        "txn_per_month_min": 15,
        "txn_per_month_max": 25,
        "amount_ranges": [
            (0.50, 5,    50),
            (0.30, 50,   200),
            (0.15, 200,  500),
            (0.05, 500,  1000),
        ],
        # active_hours: 10,11,12,13,14,15,16,17,18,19,20,21,22,23 = 14 hours
        "active_hours": list(range(10, 24)),
        "hour_weights": [3,3,5,7,8,10,10,10,10,8,7,5,5,3],  # 14 weights
        "weekend_factor": 1.2,
        "foreign_txn_probability": 0.08,
        "foreign_country_pool": ["FR","DE","ES","IT","NL"],
        "remittance_categories": {
            "rent": 0.08, "grocery": 0.30, "utility": 0.05,
            "shopping": 0.30, "transfer": 0.20,
            "subscription": 0.05, "other": 0.02,
        },
    },
    "retiree": {
        "txn_per_month_min": 5,
        "txn_per_month_max": 10,
        "amount_ranges": [
            (0.25, 20,   100),
            (0.40, 100,  400),
            (0.30, 400,  1000),
            (0.05, 1000, 2000),
        ],
        # active_hours: 9,10,11,12,13,14,15,16,17 = 9 hours
        "active_hours": list(range(9, 18)),
        "hour_weights": [8,10,10,10,8,8,6,5,4],  # 9 weights
        "weekend_factor": 0.4,
        "foreign_txn_probability": 0.02,
        "foreign_country_pool": ["FR","ES","PT"],
        "remittance_categories": {
            "rent": 0.05, "grocery": 0.30, "utility": 0.20,
            "shopping": 0.15, "transfer": 0.20,
            "subscription": 0.05, "other": 0.05,
        },
    },
    "business": {
        "txn_per_month_min": 20,
        "txn_per_month_max": 40,
        "amount_ranges": [
            (0.20, 100,   1000),
            (0.35, 1000,  5000),
            (0.30, 5000,  20000),
            (0.15, 20000, 50000),
        ],
        # active_hours: 8,9,10,11,12,13,14,15,16,17,18 = 11 hours
        "active_hours": list(range(8, 19)),
        "hour_weights": [8,10,10,10,10,8,6,5,4,3,3],  # 11 weights
        "weekend_factor": 0.2,
        "foreign_txn_probability": 0.25,
        "foreign_country_pool": ["FR","IT","ES","NL","BE","PL","DE"],
        "remittance_categories": {
            "rent": 0.05, "utility": 0.08, "salary": 0.15,
            "shopping": 0.05, "transfer": 0.05,
            "subscription": 0.07, "supplier": 0.45, "other": 0.10,
        },
    },
}

# Remittance text templates per category
# These simulate realistic payment reference text
REMITTANCE_TEMPLATES = {
    "rent":         ["Rent {month} {year}", "Miete {month}", "Loyer {month} {year}",
                     "Monthly rent payment", "Huur {month}"],
    "grocery":      ["Weekly groceries", "Supermarket", "REWE {date}",
                     "Lidl purchase", "Carrefour", "Grocery shopping"],
    "utility":      ["Electric bill {month}", "Gas {month} {year}",
                     "Internet bill", "Water {quarter}", "Utility {month}"],
    "salary":       ["Salary {month} {year}", "Gehalt {month}", "Salaire {month}",
                     "Payroll {month}", "Monthly salary"],
    "shopping":     ["Online purchase", "Amazon order", "Zalando",
                     "H&M purchase", "Online shopping {date}"],
    "transfer":     ["Transfer to {name}", "Personal transfer",
                     "Peer payment", "Split bill", "Repayment"],
    "subscription": ["Netflix", "Spotify", "Adobe subscription",
                     "Gym membership", "Software license"],
    "supplier":     ["Invoice {inv_no}", "Supplier payment {inv_no}",
                     "PO {po_no}", "Supply chain payment", "Materials {month}"],
    "other":        ["Miscellaneous", "Payment ref {ref}", "General transfer",
                     "Various expenses"],
}

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def pick_amount(amount_ranges):
    """Sample amount from persona's distribution using lognormal."""
    probs    = [r[0] for r in amount_ranges]
    selected = random.choices(amount_ranges, weights=probs, k=1)[0]
    _, lo, hi = selected
    raw = random.lognormvariate(0, 0.5)
    raw = raw / (raw + 1)
    return round(lo + raw * (hi - lo), 2)


def pick_timestamp(base_date, persona):
    """
    Generate a realistic timestamp for a transaction.
    Weighted toward persona's active hours.
    Weekends have reduced activity based on persona's weekend_factor.
    """
    # Pick hour weighted by persona's activity pattern
    hour = random.choices(
        persona["active_hours"],
        weights=persona["hour_weights"],
        k=1
    )[0]

    # Pick minute and second uniformly
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    ts = base_date.replace(hour=hour, minute=minute, second=second)
    return ts


def pick_beneficiary(account, persona, home_country, seen_beneficiaries):
    """
    Pick a beneficiary IBAN for this transaction.
    80% from known beneficiary list
    20% new (domestic or occasionally foreign)
    Returns (iban, country, is_new)
    """
    known = account["known_beneficiaries_list"]

    if known and random.random() < 0.80:
        # Pick from known beneficiaries
        iban    = random.choice(known)
        country = iban[:2] if iban[:2] != "OT" else home_country
        is_new  = iban not in seen_beneficiaries
    else:
        # Generate a new beneficiary
        if random.random() < persona["foreign_txn_probability"]:
            country = random.choice(persona["foreign_country_pool"])
        else:
            country = home_country
        iban   = generate_iban(country)
        is_new = True   # new generated IBAN = always new

    seen_beneficiaries.add(iban)
    return iban, country, is_new


def generate_iban(country_code):
    """Generate a syntactically correct fake IBAN."""
    IBAN_LENGTHS = {
        "DE": 22, "FR": 27, "IT": 27, "ES": 24,
        "NL": 18, "BE": 16, "AT": 20, "PL": 28,
        "PT": 25, "SE": 24, "RO": 24,
    }
    length       = IBAN_LENGTHS.get(country_code, 22)
    bban_length  = length - 4
    bban         = ''.join([str(random.randint(0, 9)) for _ in range(bban_length)])
    check_digits = str(random.randint(10, 99))
    return f"{country_code}{check_digits}{bban}"


def pick_remittance(category, ts):
    """Pick a realistic remittance text for the given category."""
    templates = REMITTANCE_TEMPLATES.get(category, REMITTANCE_TEMPLATES["other"])
    template  = random.choice(templates)
    return template.format(
        month   = MONTHS[ts.month - 1],
        year    = ts.year,
        date    = ts.strftime("%d/%m"),
        quarter = f"Q{(ts.month-1)//3 + 1}",
        inv_no  = f"INV-{random.randint(1000,9999)}",
        po_no   = f"PO-{random.randint(100,999)}",
        ref     = f"REF{random.randint(10000,99999)}",
        name    = random.choice(["J.Smith","M.Mueller","A.Dupont","L.Rossi"]),
    )


def classify_country(beneficiary_country, home_country):
    """Classify beneficiary country relative to sender."""
    EU_COUNTRIES = {
        "DE","FR","IT","ES","NL","BE","AT","PL",
        "PT","SE","RO","CZ","HU","GR","DK","FI",
        "IE","SK","HR","BG","LT","LV","EE","SI",
        "CY","MT","LU"
    }
    if beneficiary_country == home_country:
        return "domestic"
    elif beneficiary_country in EU_COUNTRIES:
        return "eu_cross_border"
    else:
        return "non_eu"


def generate_transaction_dates(persona, year=2024):
    """
    Generate all transaction dates for a full year for one account.
    Returns sorted list of datetime objects.
    """
    all_dates = []

    for month in range(1, 13):
        # How many transactions this month?
        n_txn = random.randint(
            persona["txn_per_month_min"],
            persona["txn_per_month_max"]
        )

        # Get all days in this month
        if month == 12:
            days_in_month = (datetime(year+1, 1, 1) - datetime(year, month, 1)).days
        else:
            days_in_month = (datetime(year, month+1, 1) - datetime(year, month, 1)).days

        # Generate n_txn dates within this month
        for _ in range(n_txn):
            day = random.randint(1, days_in_month)
            base_date = datetime(year, month, day)

            # Apply weekend factor — lower probability of transacting on weekends
            if base_date.weekday() >= 5:   # Saturday=5, Sunday=6
                if random.random() > persona["weekend_factor"]:
                    # Skip this transaction (weekend suppression)
                    day = random.randint(1, days_in_month - 2)
                    base_date = datetime(year, month, max(1, day))

            ts = pick_timestamp(base_date, persona)
            all_dates.append(ts)

    return sorted(all_dates)


# ─────────────────────────────────────────
# LOAD ACCOUNTS FROM STEP 1
# ─────────────────────────────────────────

def load_accounts(filepath):
    """Load accounts CSV from Step 1."""
    accounts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON fields
            row["known_beneficiaries_list"] = json.loads(row["known_beneficiaries"])
            row["fraud_target_list"]        = json.loads(row["fraud_target_types"])
            accounts.append(row)
    print(f"Loaded {len(accounts):,} accounts from {filepath}")
    return accounts


# ─────────────────────────────────────────
# MAIN GENERATION LOGIC
# ─────────────────────────────────────────

def generate_normal_transactions(accounts):
    """
    Generate 12 months of normal transaction history
    for all accounts. Returns list of transaction dicts.
    """
    all_transactions = []
    txn_counter      = 0

    print(f"\nGenerating normal transactions for {len(accounts):,} accounts...")
    print("(12 months per account — this may take a minute)\n")

    for idx, account in enumerate(accounts):

        persona      = PERSONAS[account["persona"]]
        home_country = account["home_country"]

        # Track which beneficiaries this account has paid before
        # Start with their known beneficiaries as "already seen"
        seen_beneficiaries = set(account["known_beneficiaries_list"])

        # Generate all transaction timestamps for the year
        transaction_dates = generate_transaction_dates(persona)

        # Track previous transaction timestamp for time_since_last_txn
        prev_timestamp = None

        for ts in transaction_dates:

            # 1. Pick beneficiary
            ben_iban, ben_country, is_new = pick_beneficiary(
                account, persona, home_country, seen_beneficiaries
            )

            # 2. Pick amount
            amount = pick_amount(persona["amount_ranges"])

            # 3. Pick remittance category and text
            categories = persona["remittance_categories"]
            category   = random.choices(
                list(categories.keys()),
                weights=list(categories.values()),
                k=1
            )[0]
            remittance_text = pick_remittance(category, ts)

            # 4. Compute engineered features
            time_since_last = None
            if prev_timestamp is not None:
                delta = (ts - prev_timestamp).total_seconds()
                time_since_last = round(delta, 0)

            country_classification = classify_country(ben_country, home_country)

            # 5. Build transaction record
            txn = {
                "transaction_id":       f"TXN_{txn_counter:08d}",
                "account_id":           account["account_id"],
                "persona":              account["persona"],
                "timestamp":            ts.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_iban":          account["sender_iban"],
                "beneficiary_iban":     ben_iban,
                "beneficiary_country":  ben_country,
                "country_type":         country_classification,
                "amount":               amount,
                "remittance_category":  category,
                "remittance_text":      remittance_text,
                "hour_of_day":          ts.hour,
                "day_of_week":          ts.weekday(),   # 0=Monday, 6=Sunday
                "is_weekend":           1 if ts.weekday() >= 5 else 0,
                "time_since_last_txn":  time_since_last,
                "is_new_beneficiary":   1 if is_new else 0,
                "is_fraud":             0,
                "fraud_type":           "none",
            }

            all_transactions.append(txn)
            prev_timestamp = ts
            txn_counter   += 1

        # Progress update every 1000 accounts
        if (idx + 1) % 1000 == 0:
            print(f"  → {idx+1:,} accounts processed | "
                  f"{txn_counter:,} transactions generated so far...")

    return all_transactions


def save_transactions(transactions, filepath):
    """Save transactions to CSV."""
    if not transactions:
        print("No transactions to save.")
        return

    fieldnames = list(transactions[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    print(f"\n✅ Saved {len(transactions):,} transactions to {filepath}")


def print_summary(transactions):
    """Print summary statistics of generated transactions."""
    from collections import Counter

    total    = len(transactions)
    personas = Counter(t["persona"] for t in transactions)
    cats     = Counter(t["remittance_category"] for t in transactions)
    new_ben  = sum(1 for t in transactions if t["is_new_beneficiary"] == 1)
    foreign  = sum(1 for t in transactions if t["country_type"] != "domestic")
    amounts  = [t["amount"] for t in transactions]
    weekend  = sum(1 for t in transactions if t["is_weekend"] == 1)

    print("\n" + "="*55)
    print("NORMAL TRANSACTION GENERATION SUMMARY")
    print("="*55)
    print(f"Total transactions:        {total:>10,}")
    print(f"New beneficiary txns:      {new_ben:>10,}  ({new_ben/total*100:.1f}%)")
    print(f"Cross-border txns:         {foreign:>10,}  ({foreign/total*100:.1f}%)")
    print(f"Weekend txns:              {weekend:>10,}  ({weekend/total*100:.1f}%)")
    print(f"Avg amount:                €{sum(amounts)/len(amounts):>9.2f}")
    print(f"Min amount:                €{min(amounts):>9.2f}")
    print(f"Max amount:                €{max(amounts):>9.2f}")

    print("\nTransactions per persona:")
    for persona, count in sorted(personas.items()):
        print(f"  {persona:<12} {count:>8,}  ({count/total*100:.1f}%)")

    print("\nTop remittance categories:")
    for cat, count in cats.most_common(5):
        print(f"  {cat:<15} {count:>8,}  ({count/total*100:.1f}%)")

    # Sample transaction
    sample = transactions[100]
    print(f"\nSample transaction (record 100):")
    print(f"  ID:             {sample['transaction_id']}")
    print(f"  Account:        {sample['account_id']} ({sample['persona']})")
    print(f"  Timestamp:      {sample['timestamp']}")
    print(f"  Amount:         €{sample['amount']}")
    print(f"  Beneficiary:    {sample['beneficiary_iban']}")
    print(f"  Country type:   {sample['country_type']}")
    print(f"  Category:       {sample['remittance_category']}")
    print(f"  New beneficiary:{sample['is_new_beneficiary']}")
    print(f"  Time since last:{sample['time_since_last_txn']}s")
    print("="*55)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Load accounts from Step 1
    accounts = load_accounts(ACCOUNTS_FILE)

    # Generate normal transactions
    transactions = generate_normal_transactions(accounts)

    # Print summary
    print_summary(transactions)

    # Save to CSV
    save_transactions(transactions, OUTPUT_FILE)
