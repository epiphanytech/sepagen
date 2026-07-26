"""
SEPAGen - Step 3: APP Fraud Injection
=======================================
Injects realistic APP fraud transactions into the normal
transaction dataset following EPC 2025 fraud typology signatures.

Fraud typologies implemented:
  1. Bank / Authority Impersonation
  2. Invoice / Mandate Fraud
  3. Romance Scam
  4. CEO / Business Email Compromise (BEC)

Target fraud rate: ~0.2% (per EBA-ECB 2025 report)

Output: synsep_full_dataset.csv  ← Final SynSEPA dataset
"""

import csv
import json
import random
from datetime import datetime, timedelta
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

RANDOM_SEED             = 42
ACCOUNTS_FILE           = "accounts.csv"
NORMAL_TXN_FILE         = "normal_transactions.csv"
OUTPUT_FILE             = "synsep_full_dataset.csv"

# Fraud volume targets (per EPC typology distribution)
FRAUD_TARGETS = {
    "impersonation": 1440,
    "invoice":        900,
    "romance":        720,
    "ceo":            540,
}

# IBAN lengths per country
IBAN_LENGTHS = {
    "DE": 22, "FR": 27, "IT": 27, "ES": 24,
    "NL": 18, "BE": 16, "AT": 20, "PL": 28,
    "PT": 25, "SE": 24, "RO": 24, "HU": 28,
    "BG": 22, "LT": 20, "UA": 29,
}

# Countries used as fraud mule destinations
# Based on EPC/Europol fraud destination patterns
MULE_COUNTRIES_DOMESTIC  = ["DE","FR","IT","ES","NL","BE"]
MULE_COUNTRIES_EU        = ["RO","BG","HU","PL","LT"]
MULE_COUNTRIES_NON_EU    = ["UA","GB","TR"]

random.seed(RANDOM_SEED)


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def generate_iban(country_code):
    """Generate a syntactically correct fake IBAN."""
    length      = IBAN_LENGTHS.get(country_code, 22)
    bban_length = length - 4
    bban        = ''.join([str(random.randint(0,9)) for _ in range(bban_length)])
    check       = str(random.randint(10, 99))
    return f"{country_code}{check}{bban}"


def classify_country(ben_country, home_country):
    """Classify beneficiary country relative to sender."""
    EU = {
        "DE","FR","IT","ES","NL","BE","AT","PL","PT","SE",
        "RO","CZ","HU","GR","DK","FI","IE","SK","HR","BG",
        "LT","LV","EE","SI","CY","MT","LU"
    }
    if ben_country == home_country:
        return "domestic"
    elif ben_country in EU:
        return "eu_cross_border"
    else:
        return "non_eu"


def fraud_txn_id(counter):
    """Generate a fraud transaction ID."""
    return f"FRD_{counter:07d}"


def parse_timestamp(ts_str):
    """Parse timestamp string to datetime."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def format_timestamp(dt):
    """Format datetime to string."""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_account_max_amount(account_txns):
    """Get the maximum normal transaction amount for an account."""
    if not account_txns:
        return 1000
    return max(float(t["amount"]) for t in account_txns)


def get_account_avg_amount(account_txns):
    """Get the average normal transaction amount for an account."""
    if not account_txns:
        return 500
    return sum(float(t["amount"]) for t in account_txns) / len(account_txns)


def get_last_timestamp(account_txns):
    """Get the last transaction timestamp for an account."""
    if not account_txns:
        return datetime(2024, 6, 15, 14, 0, 0)
    return parse_timestamp(account_txns[-1]["timestamp"])


def build_fraud_txn(
    txn_id, account, timestamp, amount,
    ben_iban, ben_country,
    remittance_category, remittance_text,
    fraud_type, prev_timestamp=None
):
    """Build a fraud transaction record with all required fields."""
    time_since_last = None
    if prev_timestamp:
        time_since_last = round(
            (timestamp - prev_timestamp).total_seconds(), 0
        )

    return {
        "transaction_id":      txn_id,
        "account_id":          account["account_id"],
        "persona":             account["persona"],
        "timestamp":           format_timestamp(timestamp),
        "sender_iban":         account["sender_iban"],
        "beneficiary_iban":    ben_iban,
        "beneficiary_country": ben_country,
        "country_type":        classify_country(ben_country, account["home_country"]),
        "amount":              round(amount, 2),
        "remittance_category": remittance_category,
        "remittance_text":     remittance_text,
        "hour_of_day":         timestamp.hour,
        "day_of_week":         timestamp.weekday(),
        "is_weekend":          1 if timestamp.weekday() >= 5 else 0,
        "time_since_last_txn": time_since_last,
        "is_new_beneficiary":  1,         # fraud almost always uses new IBAN
        "is_fraud":            1,
        "fraud_type":          fraud_type,
    }


# ─────────────────────────────────────────
# FRAUD TYPOLOGY GENERATORS
# Each follows the EPC 2025 behavioural
# signature for that typology
# ─────────────────────────────────────────

def generate_impersonation_fraud(
    account, account_txns, txn_counter
):
    """
    Bank / Authority Impersonation Fraud
    ─────────────────────────────────────
    Signature (EPC 2025):
    - Single large transaction (3-10x victim's normal max)
    - New IBAN — often domestic mule account
    - Remittance: "safe account" / "security transfer" / "urgent"
    - Happens shortly after a normal transaction (phone call context)
    - Often during daytime hours (fraudster calls while victim is available)
    - Victim: employee or retiree (most targeted per EPC)
    """
    fraud_txns = []

    max_normal  = get_account_max_amount(account_txns)
    avg_normal  = get_account_avg_amount(account_txns)

    # Large amount — 3x to 10x their normal maximum
    amount = max_normal * random.uniform(3.0, 10.0)
    amount = min(amount, 50000)   # cap at €50k

    # Mule IBAN — mostly domestic (fraudster uses local mule)
    if random.random() < 0.70:
        mule_country = random.choice(MULE_COUNTRIES_DOMESTIC)
    else:
        mule_country = random.choice(MULE_COUNTRIES_EU)
    mule_iban = generate_iban(mule_country)

    # Timing — shortly after a normal transaction (simulating phone call)
    last_ts  = get_last_timestamp(account_txns)
    # Fraud happens 30 mins to 6 hours after last normal transaction
    delay_seconds = random.randint(1800, 21600)
    fraud_ts = last_ts + timedelta(seconds=delay_seconds)

    # Keep within 2024
    if fraud_ts.year > 2024:
        fraud_ts = datetime(2024, 12, 15,
                            random.randint(10, 16),
                            random.randint(0, 59))

    # Daytime hours (fraudster calls during working hours)
    fraud_ts = fraud_ts.replace(
        hour=random.randint(9, 17),
        minute=random.randint(0, 59)
    )

    remittance_texts = [
        "Safe account transfer",
        "Security transfer urgent",
        "Fraud protection transfer",
        "Urgent account protection",
        "Temporary security holding",
        "Bank security protocol",
    ]
    remittance_text = random.choice(remittance_texts)

    fraud_txns.append(build_fraud_txn(
        txn_id             = fraud_txn_id(txn_counter),
        account            = account,
        timestamp          = fraud_ts,
        amount             = amount,
        ben_iban           = mule_iban,
        ben_country        = mule_country,
        remittance_category= "transfer",
        remittance_text    = remittance_text,
        fraud_type         = "impersonation",
        prev_timestamp     = last_ts,
    ))

    return fraud_txns


def generate_invoice_fraud(
    account, account_txns, txn_counter
):
    """
    Invoice / Mandate Fraud
    ───────────────────────
    Signature (EPC 2025):
    - Amount similar to victim's normal supplier payments (subtle!)
    - Different beneficiary IBAN but plausible beneficiary name
    - Normal-looking remittance text (invoice number)
    - Normal business hours
    - Target: business accounts only
    """
    fraud_txns = []

    # Find typical supplier payment amounts from this account
    supplier_txns = [
        t for t in account_txns
        if t.get("remittance_category") in ["supplier", "utility", "other"]
    ]

    if supplier_txns:
        # Use a similar amount to their normal supplier payments
        ref_amount = float(random.choice(supplier_txns)["amount"])
        # Add slight variation — fraudster mirrors invoice closely
        amount = ref_amount * random.uniform(0.85, 1.15)
    else:
        amount = random.uniform(2000, 15000)

    # Mule IBAN — can be domestic or EU
    if random.random() < 0.60:
        mule_country = random.choice(MULE_COUNTRIES_DOMESTIC)
    else:
        mule_country = random.choice(MULE_COUNTRIES_EU)
    mule_iban = generate_iban(mule_country)

    # Normal business hours timing
    last_ts  = get_last_timestamp(account_txns)
    # Invoice fraud during business hours, a few days after last txn
    delay_days = random.randint(3, 14)
    fraud_ts   = last_ts + timedelta(days=delay_days)

    if fraud_ts.year > 2024:
        fraud_ts = datetime(2024, 12, 10,
                            random.randint(9, 17),
                            random.randint(0, 59))

    # Business hours
    fraud_ts = fraud_ts.replace(
        hour=random.randint(9, 17),
        minute=random.randint(0, 59)
    )
    # Ensure weekday
    while fraud_ts.weekday() >= 5:
        fraud_ts += timedelta(days=1)

    # Normal looking invoice reference
    inv_num = f"INV-{random.randint(2024001, 2024999)}"
    remittance_texts = [
        f"Payment {inv_num}",
        f"Invoice settlement {inv_num}",
        f"Supplier payment {inv_num}",
        f"PO settlement {random.randint(100,999)}",
        f"Monthly supply {inv_num}",
    ]
    remittance_text = random.choice(remittance_texts)

    fraud_txns.append(build_fraud_txn(
        txn_id             = fraud_txn_id(txn_counter),
        account            = account,
        timestamp          = fraud_ts,
        amount             = amount,
        ben_iban           = mule_iban,
        ben_country        = mule_country,
        remittance_category= "supplier",
        remittance_text    = remittance_text,
        fraud_type         = "invoice",
        prev_timestamp     = last_ts,
    ))

    return fraud_txns


def generate_romance_fraud(
    account, account_txns, txn_counter
):
    """
    Romance Scam
    ────────────
    Signature (EPC 2025):
    - Multiple transactions (4-8) over several weeks
    - Same foreign IBAN each time (key distinguishing feature)
    - Escalating amounts — each 1.5-2x previous
    - Emotional remittance text
    - Cross-border — always foreign destination
    - Target: employee, student, retiree
    """
    fraud_txns = []

    # Number of transactions in the romance scam sequence
    n_payments = random.randint(4, 8)

    # Always foreign mule account — key signature
    mule_country = random.choice(MULE_COUNTRIES_EU + MULE_COUNTRIES_NON_EU)
    mule_iban    = generate_iban(mule_country)   # SAME IBAN throughout

    # Starting amount — small to build trust
    avg_normal    = get_account_avg_amount(account_txns)
    start_amount  = avg_normal * random.uniform(0.05, 0.15)
    start_amount  = max(50, min(start_amount, 300))

    # Starting timestamp — pick a random point mid-year
    # Romance scams unfold over 4-8 weeks so we start Jan-Sep
    # to ensure the full sequence fits within 2024
    last_ts     = get_last_timestamp(account_txns)
    start_month = random.randint(1, 9)
    start_day   = random.randint(1, 28)
    current_ts  = datetime(2024, start_month, start_day,
                           random.randint(10, 20),
                           random.randint(0, 59))

    # Emotional remittance texts — escalate from mild to urgent
    early_texts = [
        "Help with flight ticket",
        "Small gift for you",
        "Transfer to my account",
        "Medical consultation fee",
    ]
    mid_texts = [
        "Medical emergency help",
        "Stuck abroad need help",
        "Urgent medical bill",
        "Emergency situation",
    ]
    late_texts = [
        "Customs release fee",
        "Urgent emergency transfer",
        "Help me please urgent",
        "Last transfer I promise",
        "Investment opportunity urgent",
    ]

    amount = start_amount
    prev_ts = last_ts

    for i in range(n_payments):
        # Keep within 2024
        if current_ts.year > 2024:
            break

        # Amount escalates each payment
        escalation = random.uniform(1.4, 2.2)
        if i > 0:
            amount = amount * escalation
        amount = min(amount, 5000)   # cap individual payment

        # Pick remittance text based on stage of scam
        if i < 2:
            text = random.choice(early_texts)
        elif i < 5:
            text = random.choice(mid_texts)
        else:
            text = random.choice(late_texts)

        fraud_txns.append(build_fraud_txn(
            txn_id             = fraud_txn_id(txn_counter + i),
            account            = account,
            timestamp          = current_ts,
            amount             = amount,
            ben_iban           = mule_iban,
            ben_country        = mule_country,
            remittance_category= "transfer",
            remittance_text    = text,
            fraud_type         = "romance",
            prev_timestamp     = prev_ts,
        ))

        prev_ts    = current_ts
        # Each payment ~7-10 days apart
        current_ts = current_ts + timedelta(
            days   = random.randint(5, 12),
            hours  = random.randint(-3, 3)
        )

    return fraud_txns


def generate_ceo_fraud(
    account, account_txns, txn_counter
):
    """
    CEO / Business Email Compromise (BEC) Fraud
    ─────────────────────────────────────────────
    Signature (EPC 2025):
    - Large amount — plausible for business but unusual
    - New foreign IBAN (often non-EU)
    - Friday afternoon timing (urgency before weekend)
    - Vague / confidential remittance text
    - Single large transaction
    - Target: business accounts only
    """
    fraud_txns = []

    max_normal = get_account_max_amount(account_txns)
    avg_normal = get_account_avg_amount(account_txns)

    # Large but plausible for a business
    amount = avg_normal * random.uniform(2.0, 5.0)
    amount = max(amount, 10000)
    amount = min(amount, 200000)

    # Often non-EU destination — key signature
    if random.random() < 0.60:
        mule_country = random.choice(MULE_COUNTRIES_NON_EU)
    else:
        mule_country = random.choice(MULE_COUNTRIES_EU)
    mule_iban = generate_iban(mule_country)

    # Friday afternoon timing — CEO fraud classic pattern
    last_ts = get_last_timestamp(account_txns)

    # Find next Friday
    fraud_ts = last_ts + timedelta(days=1)
    while fraud_ts.weekday() != 4:   # 4 = Friday
        fraud_ts += timedelta(days=1)

    if fraud_ts.year > 2024:
        fraud_ts = datetime(2024, 12, 13, 15, 30, 0)   # Friday Dec 13

    # Friday afternoon: 14:00 - 17:30
    fraud_ts = fraud_ts.replace(
        hour   = random.randint(14, 17),
        minute = random.randint(0, 30)
    )

    remittance_texts = [
        "Acquisition deposit confidential",
        "Urgent wire transfer CFO approval",
        "Confidential business transfer",
        "Strategic acquisition payment",
        "Vendor payment urgent CEO approved",
        "Consulting fee confidential",
        "M&A transaction deposit",
    ]
    remittance_text = random.choice(remittance_texts)

    fraud_txns.append(build_fraud_txn(
        txn_id             = fraud_txn_id(txn_counter),
        account            = account,
        timestamp          = fraud_ts,
        amount             = amount,
        ben_iban           = mule_iban,
        ben_country        = mule_country,
        remittance_category= "transfer",
        remittance_text    = remittance_text,
        fraud_type         = "ceo",
        prev_timestamp     = last_ts,
    ))

    return fraud_txns


# ─────────────────────────────────────────
# FRAUD INJECTION ORCHESTRATOR
# ─────────────────────────────────────────

FRAUD_GENERATORS = {
    "impersonation": generate_impersonation_fraud,
    "invoice":       generate_invoice_fraud,
    "romance":       generate_romance_fraud,
    "ceo":           generate_ceo_fraud,
}

# Which personas are eligible for each fraud type
ELIGIBLE_PERSONAS = {
    "impersonation": ["employee", "retiree"],
    "invoice":       ["business"],
    "romance":       ["employee", "student", "retiree"],
    "ceo":           ["business"],
}


def inject_fraud(accounts, account_txns_map):
    """
    Inject fraud transactions into eligible accounts.
    Returns list of all fraud transactions.
    """
    all_fraud_txns = []
    txn_counter    = 0

    print("\nInjecting APP fraud transactions...")
    print(f"{'Typology':<18} {'Target':>8} {'Generated':>10}")
    print("-" * 40)

    for fraud_type, target_count in FRAUD_TARGETS.items():

        eligible_personas = ELIGIBLE_PERSONAS[fraud_type]
        generator         = FRAUD_GENERATORS[fraud_type]

        # Filter eligible accounts
        eligible_accounts = [
            a for a in accounts
            if a["persona"] in eligible_personas
            and len(account_txns_map.get(a["account_id"], [])) >= 5
        ]

        if not eligible_accounts:
            print(f"  {fraud_type:<16} No eligible accounts found!")
            continue

        # Sample victim accounts (with replacement allowed for large targets)
        n_victims = min(target_count, len(eligible_accounts))
        victims   = random.sample(eligible_accounts, n_victims)

        # If we need more than available, sample with replacement
        while len(victims) < target_count:
            victims.append(random.choice(eligible_accounts))

        fraud_txns_this_type = []

        for victim in victims[:target_count]:
            account_txns = account_txns_map.get(victim["account_id"], [])

            # Generate fraud transaction(s) for this victim
            new_fraud = generator(
                account      = victim,
                account_txns = account_txns,
                txn_counter  = txn_counter,
            )

            fraud_txns_this_type.extend(new_fraud)
            txn_counter += len(new_fraud)

        all_fraud_txns.extend(fraud_txns_this_type)
        print(f"  {fraud_type:<18} {target_count:>8,} {len(fraud_txns_this_type):>10,}")

    print("-" * 40)
    print(f"  {'TOTAL':<18} {sum(FRAUD_TARGETS.values()):>8,} {len(all_fraud_txns):>10,}")

    return all_fraud_txns


# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────

def load_accounts(filepath):
    """Load accounts from Step 1 CSV."""
    accounts = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["known_beneficiaries_list"] = json.loads(row["known_beneficiaries"])
            row["fraud_target_list"]        = json.loads(row["fraud_target_types"])
            accounts.append(row)
    print(f"Loaded {len(accounts):,} accounts")
    return accounts


def load_normal_transactions(filepath):
    """
    Load normal transactions and build a per-account index.
    Returns (all_txns list, account_txns_map dict)
    """
    print(f"Loading normal transactions from {filepath}...")
    print("(This may take ~30 seconds for 1.8M rows)\n")

    all_txns         = []
    account_txns_map = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_txns.append(row)
            account_txns_map[row["account_id"]].append(row)

    print(f"Loaded {len(all_txns):,} normal transactions")
    print(f"Indexed across {len(account_txns_map):,} accounts")
    return all_txns, account_txns_map


# ─────────────────────────────────────────
# FINAL DATASET ASSEMBLY
# ─────────────────────────────────────────

def assemble_and_save(normal_txns, fraud_txns, output_file):
    """
    Combine normal + fraud transactions,
    sort by account_id + timestamp,
    save to final CSV.
    """
    print(f"\nAssembling final dataset...")

    all_txns = normal_txns + fraud_txns

    # Sort by account_id then timestamp for proper sequence ordering
    print("Sorting by account + timestamp...")
    all_txns.sort(key=lambda t: (
        t["account_id"],
        t["timestamp"]
    ))

    # Recompute time_since_last_txn after sorting
    # (fraud txns inserted into sequence need correct delta)
    print("Recomputing time_since_last_txn after fraud injection...")
    prev_ts_map = {}

    for txn in all_txns:
        acc_id = txn["account_id"]
        curr_ts = parse_timestamp(txn["timestamp"])

        if acc_id in prev_ts_map:
            delta = (curr_ts - prev_ts_map[acc_id]).total_seconds()
            txn["time_since_last_txn"] = round(delta, 0)
        else:
            txn["time_since_last_txn"] = None

        prev_ts_map[acc_id] = curr_ts

    # Save
    print(f"Saving {len(all_txns):,} transactions to {output_file}...")
    fieldnames = list(all_txns[0].keys())

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_txns)

    print(f"\n✅ Saved final dataset: {output_file}")
    return all_txns


def print_final_summary(all_txns):
    """Print final dataset statistics."""
    total       = len(all_txns)
    fraud       = sum(1 for t in all_txns if int(t["is_fraud"]) == 1)
    normal      = total - fraud
    fraud_rate  = fraud / total * 100

    from collections import Counter
    fraud_types = Counter(
        t["fraud_type"] for t in all_txns
        if int(t["is_fraud"]) == 1
    )

    print("\n" + "="*55)
    print("SYNSEP FINAL DATASET SUMMARY")
    print("="*55)
    print(f"Total transactions:     {total:>10,}")
    print(f"Normal transactions:    {normal:>10,}  ({100-fraud_rate:.3f}%)")
    print(f"Fraud transactions:     {fraud:>10,}  ({fraud_rate:.3f}%)")
    print(f"\nFraud breakdown:")
    for ftype, count in fraud_types.most_common():
        print(f"  {ftype:<18} {count:>8,}")
    print(f"\nTarget fraud rate (EBA): 0.200%")
    print(f"Achieved fraud rate:     {fraud_rate:.3f}%")
    print("="*55)
    print("\n🎉 SynSEPA dataset generation complete!")
    print("Ready for Phase 3: Data Preparation & Tokenisation")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load accounts
    accounts = load_accounts(ACCOUNTS_FILE)

    # 2. Load normal transactions
    normal_txns, account_txns_map = load_normal_transactions(NORMAL_TXN_FILE)

    # 3. Generate and inject fraud
    fraud_txns = inject_fraud(accounts, account_txns_map)

    # 4. Assemble, sort and save final dataset
    all_txns = assemble_and_save(normal_txns, fraud_txns, OUTPUT_FILE)

    # 5. Print final summary
    print_final_summary(all_txns)
