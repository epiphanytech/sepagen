"""
SEPAGen - Step 1: Account Generation
=====================================
Generates 10,000 synthetic SEPA accounts with behavioural personas.
Each account has:
  - A unique ID and IBAN
  - A persona (employee / student / retiree / business)
  - A home country (weighted by EU payment volumes)
  - A list of known beneficiary IBANs (regular payees)
  - Behavioural parameters from their persona

Output: accounts.csv
"""

import random
import csv
import json

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

TOTAL_ACCOUNTS = 10_000
RANDOM_SEED    = 42        # for reproducibility
OUTPUT_FILE    = "accounts.csv"

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────
# COUNTRIES
# Weighted by EBA-ECB 2025 SEPA payment
# volumes (approximate distribution)
# ─────────────────────────────────────────

COUNTRIES = {
    "DE": 0.25,   # Germany      — largest SEPA market
    "FR": 0.18,   # France
    "IT": 0.15,   # Italy
    "ES": 0.12,   # Spain
    "NL": 0.08,   # Netherlands
    "BE": 0.05,   # Belgium
    "AT": 0.04,   # Austria
    "PL": 0.04,   # Poland
    "PT": 0.03,   # Portugal
    "SE": 0.03,   # Sweden
    "OTHER": 0.03 # Other EU/EEA
}

# IBAN lengths per country (total including country code + check digits)
IBAN_LENGTHS = {
    "DE": 22, "FR": 27, "IT": 27, "ES": 24,
    "NL": 18, "BE": 16, "AT": 20, "PL": 28,
    "PT": 25, "SE": 24, "OTHER": 22
}

# ─────────────────────────────────────────
# PERSONA DEFINITIONS
# Each persona defines the behavioural
# parameters of an account type
# ─────────────────────────────────────────

PERSONAS = {

    "employee": {
        "weight": 0.50,           # 50% of accounts
        "description": "Regular salaried employee, personal account",

        # Transaction frequency
        "txn_per_month_min": 8,
        "txn_per_month_max": 15,

        # Amount ranges (EUR)
        "amount_ranges": [
            (0.30, 10,   100),    # (probability, min, max) — small daily purchases
            (0.35, 100,  500),    # groceries, utilities
            (0.25, 500,  1500),   # rent, larger bills
            (0.10, 1500, 3000),   # occasional large transfer
        ],

        # Active hours — weighted toward daytime
        "active_hours": list(range(8, 21)),
        "hour_weights": [5,5,8,10,10,10,10,10,8,8,5,5,3],

        # Weekend activity (fraction of weekday activity)
        "weekend_factor": 0.6,

        # Known beneficiaries (regular payees this account pays)
        "known_beneficiary_count_min": 5,
        "known_beneficiary_count_max": 12,

        # Cross-border tendency
        "foreign_txn_probability": 0.05,  # 5% of txns go abroad
        "foreign_country_pool": ["FR","ES","IT","NL","PT"],  # EU holiday/family

        # Remittance categories and their probabilities
        "remittance_categories": {
            "rent":       0.10,
            "grocery":    0.25,
            "utility":    0.10,
            "salary":     0.00,   # receives salary, doesn't send
            "shopping":   0.20,
            "transfer":   0.15,   # peer transfer to friends/family
            "subscription": 0.10,
            "other":      0.10,
        },

        # Which fraud typologies can TARGET this persona
        "fraud_target": ["impersonation", "romance"],
    },

    "student": {
        "weight": 0.20,
        "description": "University student, personal account, low income",

        "txn_per_month_min": 15,
        "txn_per_month_max": 25,

        "amount_ranges": [
            (0.50, 5,    50),     # frequent small purchases
            (0.30, 50,   200),    # groceries, small bills
            (0.15, 200,  500),    # occasional larger expense
            (0.05, 500,  1000),   # rare larger transfer
        ],

        "active_hours": list(range(10, 24)),
        "hour_weights": [3,3,5,7,8,10,10,10,10,8,7,5,5,3],
        "weekend_factor": 1.2,   # students more active on weekends

        "known_beneficiary_count_min": 3,
        "known_beneficiary_count_max": 8,

        "foreign_txn_probability": 0.08,  # Erasmus, travel
        "foreign_country_pool": ["FR","DE","ES","IT","NL"],

        "remittance_categories": {
            "rent":       0.08,
            "grocery":    0.30,
            "utility":    0.05,
            "salary":     0.00,
            "shopping":   0.30,
            "transfer":   0.20,
            "subscription": 0.05,
            "other":      0.02,
        },

        "fraud_target": ["romance", "impersonation"],
    },

    "retiree": {
        "weight": 0.15,
        "description": "Retired individual, predictable low-variance behaviour",

        "txn_per_month_min": 5,
        "txn_per_month_max": 10,

        "amount_ranges": [
            (0.25, 20,   100),
            (0.40, 100,  400),
            (0.30, 400,  1000),
            (0.05, 1000, 2000),
        ],

        "active_hours": list(range(9, 18)),
        "hour_weights": [8,10,10,10,8,8,6,5,4,3],
        "weekend_factor": 0.4,   # retirees less active weekends

        "known_beneficiary_count_min": 4,
        "known_beneficiary_count_max": 8,

        "foreign_txn_probability": 0.02,  # very rarely sends abroad
        "foreign_country_pool": ["FR","ES","PT"],

        "remittance_categories": {
            "rent":       0.05,
            "grocery":    0.30,
            "utility":    0.20,
            "salary":     0.00,
            "shopping":   0.15,
            "transfer":   0.20,   # family transfers
            "subscription": 0.05,
            "other":      0.05,
        },

        # Retirees are most vulnerable to impersonation
        "fraud_target": ["impersonation", "romance"],
    },

    "business": {
        "weight": 0.15,
        "description": "Small business owner, higher value irregular payments",

        "txn_per_month_min": 20,
        "txn_per_month_max": 40,

        "amount_ranges": [
            (0.20, 100,   1000),   # small operational costs
            (0.35, 1000,  5000),   # supplier payments
            (0.30, 5000,  20000),  # larger supplier/payroll
            (0.15, 20000, 50000),  # major payments
        ],

        "active_hours": list(range(8, 19)),
        "hour_weights": [8,10,10,10,10,8,6,5,4,3,3,3],
        "weekend_factor": 0.2,   # businesses mostly inactive weekends

        "known_beneficiary_count_min": 10,
        "known_beneficiary_count_max": 20,

        "foreign_txn_probability": 0.25,  # frequent EU supplier payments
        "foreign_country_pool": ["FR","IT","ES","NL","BE","PL","DE"],

        "remittance_categories": {
            "rent":         0.05,
            "grocery":      0.00,
            "utility":      0.08,
            "salary":       0.15,   # payroll
            "shopping":     0.05,
            "transfer":     0.05,
            "subscription": 0.07,
            "supplier":     0.45,   # main category for business
            "other":        0.10,
        },

        # Business accounts targeted by invoice and CEO fraud
        "fraud_target": ["invoice", "ceo"],
    },
}

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def weighted_choice(options_dict):
    """Pick a key from a dict of {key: weight}."""
    keys   = list(options_dict.keys())
    weights = list(options_dict.values())
    return random.choices(keys, weights=weights, k=1)[0]


def generate_iban(country_code):
    """
    Generate a syntactically correct but fake IBAN.
    Format: CC + 2 check digits + BBAN (numeric)
    Real IBANs have checksum validation — we keep format correct
    but don't compute real Mod-97 checksum (not needed for ML training).
    """
    if country_code == "OTHER":
        country_code = "RO"   # Romania as representative non-major EU
    length = IBAN_LENGTHS.get(country_code, 22)
    bban_length = length - 4   # subtract country code (2) + check digits (2)
    bban = ''.join([str(random.randint(0, 9)) for _ in range(bban_length)])
    check_digits = str(random.randint(10, 99))
    return f"{country_code}{check_digits}{bban}"


def pick_amount(amount_ranges):
    """
    Pick a transaction amount from persona's range distribution.
    amount_ranges = list of (probability, min, max) tuples
    """
    probs    = [r[0] for r in amount_ranges]
    selected = random.choices(amount_ranges, weights=probs, k=1)[0]
    _, lo, hi = selected
    # Lognormal distribution within the range for realism
    # (most transactions cluster toward lower end of range)
    raw = random.lognormvariate(0, 0.5)
    raw = raw / (raw + 1)    # squeeze to 0-1
    amount = lo + raw * (hi - lo)
    return round(amount, 2)


def generate_known_beneficiaries(persona, home_country):
    """
    Generate a list of IBANs this account regularly pays.
    Mix of domestic and occasional foreign beneficiaries.
    """
    count = random.randint(
        persona["known_beneficiary_count_min"],
        persona["known_beneficiary_count_max"]
    )
    beneficiaries = []
    for _ in range(count):
        # 90% domestic, 10% foreign known beneficiaries
        if random.random() < 0.10 and persona["foreign_txn_probability"] > 0:
            country = random.choice(persona["foreign_country_pool"])
        else:
            country = home_country
        beneficiaries.append(generate_iban(country))
    return beneficiaries


# ─────────────────────────────────────────
# MAIN GENERATION LOGIC
# ─────────────────────────────────────────

def generate_accounts(total=TOTAL_ACCOUNTS):
    """Generate all accounts and return as list of dicts."""
    
    # Build persona selection weights
    persona_names   = list(PERSONAS.keys())
    persona_weights = [PERSONAS[p]["weight"] for p in persona_names]

    accounts = []

    print(f"Generating {total:,} accounts...")

    for i in range(total):
        # 1. Assign persona
        persona_name = random.choices(persona_names, weights=persona_weights, k=1)[0]
        persona      = PERSONAS[persona_name]

        # 2. Assign home country
        home_country = weighted_choice(COUNTRIES)

        # 3. Generate sender IBAN
        sender_iban = generate_iban(home_country)

        # 4. Generate known beneficiary IBANs
        known_beneficiaries = generate_known_beneficiaries(persona, home_country)

        # 5. Sample a typical amount for this account (their "normal" range)
        typical_amount = pick_amount(persona["amount_ranges"])

        # 6. Build account record
        account = {
            "account_id":            f"ACC_{i:06d}",
            "persona":               persona_name,
            "description":           persona["description"],
            "home_country":          home_country,
            "sender_iban":           sender_iban,
            "txn_per_month_min":     persona["txn_per_month_min"],
            "txn_per_month_max":     persona["txn_per_month_max"],
            "typical_amount":        typical_amount,
            "foreign_txn_prob":      persona["foreign_txn_probability"],
            "weekend_factor":        persona["weekend_factor"],
            "fraud_target_types":    json.dumps(persona["fraud_target"]),
            "known_beneficiaries":   json.dumps(known_beneficiaries),
            "known_beneficiary_count": len(known_beneficiaries),
        }

        accounts.append(account)

        # Progress update every 2000 accounts
        if (i + 1) % 2000 == 0:
            print(f"  → {i+1:,} accounts generated...")

    return accounts


def save_accounts(accounts, filepath):
    """Save accounts to CSV."""
    if not accounts:
        print("No accounts to save.")
        return

    fieldnames = list(accounts[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accounts)

    print(f"\n✅ Saved {len(accounts):,} accounts to {filepath}")


def print_summary(accounts):
    """Print a summary of generated accounts."""
    from collections import Counter

    persona_counts  = Counter(a["persona"] for a in accounts)
    country_counts  = Counter(a["home_country"] for a in accounts)

    print("\n" + "="*50)
    print("ACCOUNT GENERATION SUMMARY")
    print("="*50)
    print(f"Total accounts: {len(accounts):,}")

    print("\nPersona breakdown:")
    for persona, count in sorted(persona_counts.items()):
        pct = count / len(accounts) * 100
        print(f"  {persona:<12} {count:>6,}  ({pct:.1f}%)")

    print("\nTop 5 countries:")
    for country, count in country_counts.most_common(5):
        pct = count / len(accounts) * 100
        print(f"  {country:<8} {count:>6,}  ({pct:.1f}%)")

    # Sample account
    sample = accounts[0]
    print(f"\nSample account (first record):")
    print(f"  ID:          {sample['account_id']}")
    print(f"  Persona:     {sample['persona']}")
    print(f"  Country:     {sample['home_country']}")
    print(f"  IBAN:        {sample['sender_iban']}")
    print(f"  Known payees:{sample['known_beneficiary_count']}")
    print(f"  Typical amt: €{sample['typical_amount']}")
    print("="*50)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    accounts = generate_accounts(TOTAL_ACCOUNTS)
    print_summary(accounts)
    save_accounts(accounts, OUTPUT_FILE)
