"""
SEPAGen - Step 4: Dataset Validation
======================================
Validates SynSEPA statistical properties against:
  - EBA-ECB 2025 fraud report benchmarks
  - Expected persona behavioural profiles
  - Internal consistency checks
  - Fraud typology signature verification

Produces a validation report: validation_report.txt
"""

import csv
import json
from collections import defaultdict, Counter
from datetime import datetime
import statistics
import math

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DATASET_FILE    = "synsep_full_dataset.csv"
ACCOUNTS_FILE   = "accounts.csv"
REPORT_FILE     = "validation_report.txt"

# EBA-ECB 2025 benchmarks to validate against
EBA_BENCHMARKS = {
    "fraud_rate_pct":         0.200,   # 0.002% of value ~ 0.2% of volume (approx)
    "cross_border_pct":       11.0,    # ~11% cross-border for retail
    "weekend_pct_max":        25.0,    # weekends should be <25% of transactions
    "new_beneficiary_pct":    20.0,    # ~20% new beneficiaries in normal txns
}

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

def load_dataset(filepath):
    """Load full dataset."""
    print(f"Loading dataset from {filepath}...")
    txns = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            txns.append(row)
    print(f"Loaded {len(txns):,} transactions\n")
    return txns


def load_accounts(filepath):
    """Load accounts."""
    accounts = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accounts[row["account_id"]] = row
    return accounts


# ─────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────

class ValidationReport:
    """Collects and formats validation results."""

    def __init__(self):
        self.results  = []
        self.passed   = 0
        self.failed   = 0
        self.warnings = 0

    def check(self, name, condition, actual, expected, unit="", warn_only=False):
        status = "✅ PASS" if condition else ("⚠️  WARN" if warn_only else "❌ FAIL")
        if condition:
            self.passed += 1
        elif warn_only:
            self.warnings += 1
        else:
            self.failed += 1

        line = f"{status} | {name:<45} | actual: {actual:<12} | expected: {expected} {unit}"
        self.results.append(line)
        print(line)

    def section(self, title):
        line = f"\n{'─'*80}\n  {title}\n{'─'*80}"
        self.results.append(line)
        print(line)

    def note(self, text):
        line = f"       {text}"
        self.results.append(line)
        print(line)

    def summary(self):
        total = self.passed + self.failed + self.warnings
        lines = [
            f"\n{'='*80}",
            f"  VALIDATION SUMMARY",
            f"{'='*80}",
            f"  Total checks:  {total}",
            f"  ✅ Passed:     {self.passed}",
            f"  ❌ Failed:     {self.failed}",
            f"  ⚠️  Warnings:  {self.warnings}",
            f"{'='*80}",
        ]
        for l in lines:
            print(l)
        self.results.extend(lines)

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("SEPAGen — SynSEPA Dataset Validation Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            for line in self.results:
                f.write(line + "\n")
        print(f"\n📄 Validation report saved to {filepath}")


# ─────────────────────────────────────────
# INDIVIDUAL VALIDATION FUNCTIONS
# ─────────────────────────────────────────

def validate_basic_counts(txns, report):
    """Check basic dataset size and composition."""
    report.section("1. BASIC COUNTS AND COMPOSITION")

    total   = len(txns)
    fraud   = sum(1 for t in txns if t["is_fraud"] == "1")
    normal  = total - fraud
    accounts = len(set(t["account_id"] for t in txns))

    report.check("Total transactions > 1M",
                 total > 1_000_000,
                 f"{total:,}", "> 1,000,000")

    report.check("Unique accounts = 10,000",
                 accounts == 10000,
                 f"{accounts:,}", "10,000")

    report.check("Normal transactions > 99%",
                 normal/total*100 > 99.0,
                 f"{normal/total*100:.3f}%", "> 99.0%")

    fraud_rate = fraud/total*100
    report.check("Fraud rate 0.1% - 0.5%",
                 0.10 <= fraud_rate <= 0.50,
                 f"{fraud_rate:.3f}%", "0.1% - 0.5%")

    report.note(f"EBA benchmark fraud rate: ~0.200% | Achieved: {fraud_rate:.3f}%")

    return fraud_rate


def validate_fraud_typologies(txns, report):
    """Check fraud typology distribution."""
    report.section("2. FRAUD TYPOLOGY DISTRIBUTION")

    fraud_txns   = [t for t in txns if t["is_fraud"] == "1"]
    type_counts  = Counter(t["fraud_type"] for t in fraud_txns)
    total_fraud  = len(fraud_txns)

    expected_types = ["impersonation", "invoice", "romance", "ceo"]
    for ft in expected_types:
        count = type_counts.get(ft, 0)
        pct   = count / total_fraud * 100 if total_fraud > 0 else 0
        report.check(
            f"Fraud type '{ft}' present",
            count > 0,
            f"{count:,} ({pct:.1f}%)", "> 0"
        )

    report.check("All 4 fraud types present",
                 len([ft for ft in expected_types if type_counts.get(ft,0) > 0]) == 4,
                 f"{len(type_counts)} types", "4 types")

    # Romance scam should have multiple txns per victim
    romance_per_victim = type_counts.get("romance", 0) / 720
    report.check("Romance scam avg > 3 txns per victim",
                 romance_per_victim >= 3.0,
                 f"{romance_per_victim:.1f} txns/victim", ">= 3.0",
                 warn_only=True)


def validate_persona_behaviour(txns, accounts, report):
    """Check each persona shows realistic behavioural patterns."""
    report.section("3. PERSONA BEHAVIOURAL VALIDATION")

    # Group transactions by persona (normal only)
    normal_txns = [t for t in txns if t["is_fraud"] == "0"]
    persona_txns = defaultdict(list)
    for t in normal_txns:
        persona_txns[t["persona"]].append(t)

    # Expected persona properties
    expected = {
        "employee": {
            "avg_amount_max": 5000,
            "avg_amount_min": 50,
            "foreign_pct_max": 15.0,
        },
        "student": {
            "avg_amount_max": 500,
            "avg_amount_min": 5,
            "foreign_pct_max": 20.0,
        },
        "retiree": {
            "avg_amount_max": 2000,
            "avg_amount_min": 20,
            "foreign_pct_max": 10.0,
        },
        "business": {
            "avg_amount_max": 100000,
            "avg_amount_min": 100,
            "foreign_pct_max": 50.0,
        },
    }

    for persona, exp in expected.items():
        ptxns = persona_txns[persona]
        if not ptxns:
            report.check(f"{persona}: transactions exist", False,
                         "0", "> 0")
            continue

        amounts     = [float(t["amount"]) for t in ptxns]
        avg_amount  = statistics.mean(amounts)
        foreign_pct = sum(1 for t in ptxns
                          if t["country_type"] != "domestic") / len(ptxns) * 100

        report.check(
            f"{persona}: avg amount in range",
            exp["avg_amount_min"] <= avg_amount <= exp["avg_amount_max"],
            f"€{avg_amount:,.0f}",
            f"€{exp['avg_amount_min']:,} - €{exp['avg_amount_max']:,}"
        )

        report.check(
            f"{persona}: foreign txn % reasonable",
            foreign_pct <= exp["foreign_pct_max"],
            f"{foreign_pct:.1f}%",
            f"<= {exp['foreign_pct_max']}%"
        )


def validate_temporal_patterns(txns, report):
    """Check timing and temporal patterns are realistic."""
    report.section("4. TEMPORAL PATTERN VALIDATION")

    normal_txns = [t for t in txns if t["is_fraud"] == "0"]
    total_n     = len(normal_txns)

    # Weekend percentage
    weekend_pct = sum(1 for t in normal_txns
                      if t["is_weekend"] == "1") / total_n * 100
    report.check("Weekend txns < 25% (normal)",
                 weekend_pct < 25.0,
                 f"{weekend_pct:.1f}%", "< 25.0%")

    # Hour distribution — peak should be business hours
    hours = [int(t["hour_of_day"]) for t in normal_txns]
    business_hours = sum(1 for h in hours if 8 <= h <= 18)
    business_pct   = business_hours / len(hours) * 100
    report.check("Business hours (8-18) > 60% of txns",
                 business_pct > 60.0,
                 f"{business_pct:.1f}%", "> 60.0%")

    # time_since_last_txn — check it exists and is reasonable
    deltas = [float(t["time_since_last_txn"])
              for t in normal_txns
              if t["time_since_last_txn"] not in ("", "None", None)]

    if deltas:
        avg_delta_hours = statistics.mean(deltas) / 3600
        report.check("Avg time between txns > 1 hour",
                     avg_delta_hours > 1.0,
                     f"{avg_delta_hours:.1f} hours", "> 1 hour")

        report.check("Avg time between txns < 200 hours",
                     avg_delta_hours < 200.0,
                     f"{avg_delta_hours:.1f} hours", "< 200 hours")

    # Date range — all in 2024
    timestamps = [t["timestamp"][:4] for t in txns]
    years      = Counter(timestamps)
    report.check("All transactions in 2024",
                 all(y == "2024" for y in years.keys()),
                 str(dict(years)), "{'2024': all}")


def validate_fraud_signatures(txns, report):
    """
    Check each fraud typology shows its expected
    behavioural signature in the data.
    """
    report.section("5. FRAUD TYPOLOGY SIGNATURE VERIFICATION")

    fraud_by_type = defaultdict(list)
    for t in txns:
        if t["is_fraud"] == "1":
            fraud_by_type[t["fraud_type"]].append(t)

    normal_txns = [t for t in txns if t["is_fraud"] == "0"]

    # Get normal transaction stats for comparison
    normal_amounts     = [float(t["amount"]) for t in normal_txns]
    normal_avg_amount  = statistics.mean(normal_amounts)

    # ── Impersonation ──
    imp = fraud_by_type.get("impersonation", [])
    if imp:
        imp_amounts = [float(t["amount"]) for t in imp]
        imp_avg     = statistics.mean(imp_amounts)
        imp_new_ben = sum(1 for t in imp
                          if t["is_new_beneficiary"] == "1") / len(imp) * 100

        report.check("Impersonation: avg amount > 3x normal avg",
                     imp_avg > normal_avg_amount * 3,
                     f"€{imp_avg:,.0f}", f"> €{normal_avg_amount*3:,.0f}")

        report.check("Impersonation: new beneficiary > 95%",
                     imp_new_ben > 95.0,
                     f"{imp_new_ben:.1f}%", "> 95.0%")

        imp_daytime = sum(1 for t in imp
                          if 9 <= int(t["hour_of_day"]) <= 17) / len(imp) * 100
        report.check("Impersonation: daytime (9-17) > 70%",
                     imp_daytime > 70.0,
                     f"{imp_daytime:.1f}%", "> 70.0%",
                     warn_only=True)

    # ── Invoice ──
    inv = fraud_by_type.get("invoice", [])
    if inv:
        inv_new_ben = sum(1 for t in inv
                          if t["is_new_beneficiary"] == "1") / len(inv) * 100
        inv_persona = Counter(t["persona"] for t in inv)

        report.check("Invoice fraud: new beneficiary > 90%",
                     inv_new_ben > 90.0,
                     f"{inv_new_ben:.1f}%", "> 90.0%")

        report.check("Invoice fraud: targets business only",
                     inv_persona.get("business", 0) == len(inv),
                     f"business={inv_persona.get('business',0)}/{len(inv)}",
                     "100% business")

    # ── Romance ──
    rom = fraud_by_type.get("romance", [])
    if rom:
        rom_foreign = sum(1 for t in rom
                          if t["country_type"] != "domestic") / len(rom) * 100
        rom_new_ben = sum(1 for t in rom
                          if t["is_new_beneficiary"] == "1") / len(rom) * 100

        report.check("Romance scam: cross-border > 95%",
                     rom_foreign > 95.0,
                     f"{rom_foreign:.1f}%", "> 95.0%")

        report.check("Romance scam: new beneficiary > 90%",
                     rom_new_ben > 90.0,
                     f"{rom_new_ben:.1f}%", "> 90.0%")

    # ── CEO ──
    ceo = fraud_by_type.get("ceo", [])
    if ceo:
        ceo_amounts  = [float(t["amount"]) for t in ceo]
        ceo_avg      = statistics.mean(ceo_amounts)
        ceo_persona  = Counter(t["persona"] for t in ceo)
        ceo_friday   = sum(1 for t in ceo
                           if int(t["day_of_week"]) == 4) / len(ceo) * 100

        report.check("CEO fraud: targets business only",
                     ceo_persona.get("business", 0) == len(ceo),
                     f"business={ceo_persona.get('business',0)}/{len(ceo)}",
                     "100% business")

        report.check("CEO fraud: avg amount > normal avg",
                     ceo_avg > normal_avg_amount,
                     f"€{ceo_avg:,.0f}", f"> €{normal_avg_amount:,.0f}")

        report.check("CEO fraud: Friday > 50%",
                     ceo_friday > 50.0,
                     f"{ceo_friday:.1f}%", "> 50.0%")


def validate_cross_border(txns, report):
    """Check cross-border transaction rates."""
    report.section("6. CROSS-BORDER TRANSACTION VALIDATION")

    normal_txns = [t for t in txns if t["is_fraud"] == "0"]
    total_n     = len(normal_txns)

    domestic    = sum(1 for t in normal_txns if t["country_type"] == "domestic")
    eu_cross    = sum(1 for t in normal_txns if t["country_type"] == "eu_cross_border")
    non_eu      = sum(1 for t in normal_txns if t["country_type"] == "non_eu")

    domestic_pct = domestic / total_n * 100
    eu_pct       = eu_cross / total_n * 100
    non_eu_pct   = non_eu   / total_n * 100
    foreign_pct  = (eu_cross + non_eu) / total_n * 100

    report.check("Domestic transactions > 80%",
                 domestic_pct > 80.0,
                 f"{domestic_pct:.1f}%", "> 80.0%")

    report.check("Cross-border txns 5-20%",
                 5.0 <= foreign_pct <= 20.0,
                 f"{foreign_pct:.1f}%", "5-20%")

    report.note(f"Domestic: {domestic_pct:.1f}% | "
                f"EU cross-border: {eu_pct:.1f}% | "
                f"Non-EU: {non_eu_pct:.1f}%")

    # Fraud should have higher cross-border rate
    fraud_txns   = [t for t in txns if t["is_fraud"] == "1"]
    fraud_foreign = sum(1 for t in fraud_txns
                        if t["country_type"] != "domestic") / len(fraud_txns) * 100

    report.check("Fraud has higher cross-border rate than normal",
                 fraud_foreign > foreign_pct,
                 f"fraud={fraud_foreign:.1f}% vs normal={foreign_pct:.1f}%",
                 "fraud > normal")


def validate_sequence_integrity(txns, report):
    """Check transaction sequences are ordered correctly per account."""
    report.section("7. SEQUENCE INTEGRITY CHECK")

    # Group by account
    account_txns = defaultdict(list)
    for t in txns:
        account_txns[t["account_id"]].append(t["timestamp"])

    # Check ordering — sample 500 accounts
    sample_accounts = list(account_txns.keys())[:500]
    out_of_order    = 0

    for acc_id in sample_accounts:
        timestamps = account_txns[acc_id]
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                out_of_order += 1
                break

    report.check("Transactions ordered by timestamp (sample 500 accounts)",
                 out_of_order == 0,
                 f"{out_of_order} accounts with ordering issues",
                 "0 issues")

    # Check all accounts have at least 5 transactions
    min_txns = min(len(v) for v in account_txns.values())
    max_txns = max(len(v) for v in account_txns.values())
    avg_txns = sum(len(v) for v in account_txns.values()) / len(account_txns)

    report.check("All accounts have >= 5 transactions",
                 min_txns >= 5,
                 f"min={min_txns}", ">= 5")

    report.note(f"Transactions per account — "
                f"min: {min_txns} | max: {max_txns} | avg: {avg_txns:.0f}")

    # Check no missing required fields
    required_fields = [
        "transaction_id", "account_id", "timestamp",
        "amount", "is_fraud", "fraud_type",
        "is_new_beneficiary", "hour_of_day", "day_of_week"
    ]
    missing = 0
    for t in txns[:10000]:   # sample first 10k
        for field in required_fields:
            if field not in t or t[field] == "":
                missing += 1

    report.check("No missing required fields (sample 10k txns)",
                 missing == 0,
                 f"{missing} missing values", "0")


def validate_amount_distributions(txns, report):
    """Check amount distributions are realistic per persona."""
    report.section("8. AMOUNT DISTRIBUTION VALIDATION")

    normal_txns  = [t for t in txns if t["is_fraud"] == "0"]
    all_amounts  = [float(t["amount"]) for t in normal_txns]
    total_n      = len(all_amounts)

    # Overall
    avg    = statistics.mean(all_amounts)
    median = statistics.median(all_amounts)
    stdev  = statistics.stdev(all_amounts)

    report.note(f"Overall amounts — avg: €{avg:,.0f} | "
                f"median: €{median:,.0f} | stdev: €{stdev:,.0f}")

    # Median should be much lower than mean (right-skewed distribution)
    report.check("Distribution right-skewed (median < mean)",
                 median < avg,
                 f"median €{median:,.0f} < mean €{avg:,.0f}",
                 "median < mean")

    # No negative amounts
    negatives = sum(1 for a in all_amounts if a <= 0)
    report.check("No zero or negative amounts",
                 negatives == 0,
                 f"{negatives} invalid amounts", "0")

    # Amount buckets
    micro    = sum(1 for a in all_amounts if a < 50)    / total_n * 100
    small    = sum(1 for a in all_amounts if 50 <= a < 500) / total_n * 100
    medium   = sum(1 for a in all_amounts if 500 <= a < 5000) / total_n * 100
    large    = sum(1 for a in all_amounts if a >= 5000) / total_n * 100

    report.note(f"Amount buckets — "
                f"<€50: {micro:.1f}% | "
                f"€50-500: {small:.1f}% | "
                f"€500-5k: {medium:.1f}% | "
                f">€5k: {large:.1f}%")

    report.check("Small transactions (< €500) > 40%",
                 (micro + small) > 40.0,
                 f"{micro+small:.1f}%", "> 40.0%")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*80)
    print("  SEPAGen — SynSEPA Dataset Validation")
    print("="*80)

    # Load data
    txns     = load_dataset(DATASET_FILE)
    accounts = load_accounts(ACCOUNTS_FILE)

    # Run all validation checks
    report = ValidationReport()

    validate_basic_counts(txns, report)
    validate_fraud_typologies(txns, report)
    validate_persona_behaviour(txns, accounts, report)
    validate_temporal_patterns(txns, report)
    validate_fraud_signatures(txns, report)
    validate_cross_border(txns, report)
    validate_sequence_integrity(txns, report)
    validate_amount_distributions(txns, report)

    # Print and save summary
    report.summary()
    report.save(REPORT_FILE)
