"""
SynSEPA — Quick Start Baseline
================================
This script shows how to load SynSEPA and run a simple
Isolation Forest baseline for APP fraud detection.

This is the recommended starting point for researchers
benchmarking against SynSEPA.

Requirements:
    pip install pandas scikit-learn matplotlib

Usage:
    python sample_baseline.py
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────

print("Loading SynSEPA dataset...")
df = pd.read_csv("synsep_full_dataset.csv")
print(f"Loaded {len(df):,} transactions")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%\n")

# ─────────────────────────────────────────
# STEP 2 — TIME-BASED TRAIN/TEST SPLIT
# IMPORTANT: Never use random split for financial time series
# Train on older data, test on newer data
# ─────────────────────────────────────────

print("Splitting dataset by time...")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Training: Jan-Sep 2024 (normal transactions only)
# Validation: Oct 2024
# Test: Nov-Dec 2024 (contains fraud)

train_mask = (df["timestamp"] < "2024-10-01") & (df["is_fraud"] == 0)
val_mask   = (df["timestamp"] >= "2024-10-01") & (df["timestamp"] < "2024-11-01")
test_mask  = (df["timestamp"] >= "2024-11-01")

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

print(f"Train set:      {len(train_df):>10,} transactions (normal only)")
print(f"Validation set: {len(val_df):>10,} transactions")
print(f"Test set:       {len(test_df):>10,} transactions "
      f"({test_df['is_fraud'].sum()} fraud)")

# ─────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# Convert categorical fields to numeric
# ─────────────────────────────────────────

def encode_features(df):
    """Encode features for ML model."""
    features = df.copy()

    # Encode country type
    country_map = {"domestic": 0, "eu_cross_border": 1, "non_eu": 2}
    features["country_type_enc"] = features["country_type"].map(country_map).fillna(0)

    # Encode remittance category
    categories = features["remittance_category"].unique()
    cat_map    = {c: i for i, c in enumerate(sorted(categories))}
    features["remittance_enc"] = features["remittance_category"].map(cat_map).fillna(0)

    # Log-transform amount (reduces skew)
    features["amount_log"] = np.log1p(features["amount"].astype(float))

    # Fill missing time_since_last_txn with median
    median_delta = features["time_since_last_txn"].astype(float).median()
    features["time_since_last_txn"] = (
        features["time_since_last_txn"]
        .astype(float)
        .fillna(median_delta)
    )

    # Select feature columns for model
    feature_cols = [
        "amount_log",
        "country_type_enc",
        "remittance_enc",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "time_since_last_txn",
        "is_new_beneficiary",
    ]

    return features[feature_cols].astype(float)


print("\nEncoding features...")
X_train = encode_features(train_df)
X_val   = encode_features(val_df)
X_test  = encode_features(test_df)

y_test  = test_df["is_fraud"].astype(int).values
print(f"Feature matrix shape: {X_train.shape}")

# ─────────────────────────────────────────
# STEP 4 — ISOLATION FOREST BASELINE
# Standard unsupervised anomaly detection
# ─────────────────────────────────────────

print("\nTraining Isolation Forest baseline...")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        classification_report
    )

    # Train on normal transactions only (unsupervised)
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.004,   # estimated fraud rate
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)

    # Score test set — lower score = more anomalous
    test_scores = iso_forest.decision_function(X_test)
    # Convert: lower score → higher fraud probability
    fraud_scores = -test_scores

    # Evaluate
    auroc = roc_auc_score(y_test, fraud_scores)
    auprc = average_precision_score(y_test, fraud_scores)

    # Threshold at 1% FPR (operationally realistic for banks)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, fraud_scores)
    idx_1pct    = np.argmin(np.abs(fpr - 0.01))
    threshold   = thresholds[idx_1pct]
    predictions = (fraud_scores >= threshold).astype(int)

    print("\n" + "="*50)
    print("ISOLATION FOREST BASELINE RESULTS")
    print("="*50)
    print(f"AUROC:          {auroc:.4f}")
    print(f"PR-AUC:         {auprc:.4f}")
    print(f"Threshold @1% FPR: {threshold:.4f}")
    print(f"\nClassification Report @ 1% FPR threshold:")
    print(classification_report(y_test, predictions,
                                target_names=["Normal", "Fraud"]))
    print("="*50)
    print("\n✅ Baseline complete — use these numbers to compare against your model")

except ImportError:
    print("scikit-learn not available — install with: pip install scikit-learn")


# ─────────────────────────────────────────
# STEP 5 — QUICK DATA EXPLORATION
# ─────────────────────────────────────────

print("\n" + "="*50)
print("DATASET EXPLORATION")
print("="*50)

# Fraud rate by typology
fraud_df = df[df["is_fraud"] == 1]
print("\nFraud by typology:")
print(fraud_df["fraud_type"].value_counts())

# Amount comparison
print(f"\nAvg amount — Normal: €{df[df['is_fraud']==0]['amount'].mean():,.0f}")
print(f"Avg amount — Fraud:  €{df[df['is_fraud']==1]['amount'].mean():,.0f}")

# New beneficiary rate
print(f"\nNew beneficiary rate:")
print(f"  Normal: {df[df['is_fraud']==0]['is_new_beneficiary'].mean()*100:.1f}%")
print(f"  Fraud:  {df[df['is_fraud']==1]['is_new_beneficiary'].mean()*100:.1f}%")

# Cross-border rate
print(f"\nCross-border rate:")
normal_foreign = (df[df['is_fraud']==0]['country_type'] != 'domestic').mean()*100
fraud_foreign  = (df[df['is_fraud']==1]['country_type'] != 'domestic').mean()*100
print(f"  Normal: {normal_foreign:.1f}%")
print(f"  Fraud:  {fraud_foreign:.1f}%")

print("\n📊 Dataset ready for model training.")
print("Next step: implement your generative model and compare against this baseline.")
