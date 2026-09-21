"""
generate_sample_data.py
Extracts a curated, stratified 100-record sample dataset from merged_clean.parquet
spanning all 6 departments, 4 insurance providers, 3 risk levels, and 3 claim statuses.
Precomputes Model A and Model B predictions and stores the dataset in data/sample_dataset.json and .csv.
"""

import json
import os
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PARQUET_PATHS = [
    BASE_DIR / "Phases" / "merged_clean.parquet",
    BASE_DIR / "merged_clean.parquet"
]

MODEL_A_PATHS = [
    BASE_DIR / "Phases" / "models" / "model_a_xgboost_tuned.pkl",
    BASE_DIR / "models" / "model_a_xgboost_tuned.pkl",
    BASE_DIR / "models" / "model_a_xgboost.pkl"
]

MODEL_B_PATHS = [
    BASE_DIR / "Phases" / "models" / "model_b_tuned.pkl",
    BASE_DIR / "models" / "model_b_tuned.pkl"
]

def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

parquet_path = find_first_existing(PARQUET_PATHS)
if not parquet_path:
    raise FileNotFoundError("Could not find merged_clean.parquet")

print(f"Reading dataset from {parquet_path}...")
df = pd.read_parquet(parquet_path)

# Ensure datetime types
df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
df["billing_date"] = pd.to_datetime(df["billing_date"], errors="coerce")
df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")

# Feature engineering matching Phase 4
df["visit_frequency"] = df.groupby("patient_id")["visit_id"].transform("count")
df["avg_los_per_patient"] = df.groupby("patient_id")["length_of_stay_hours"].transform("mean").round(2)
df["days_since_registration"] = (df["visit_date"] - df["registration_date"]).dt.days

prov_rej_map = (df.groupby("insurance_provider")["claim_status"]
                .apply(lambda s: round((s == "Rejected").sum() / len(s), 4))).to_dict()
df["provider_rejection_rate"] = df["insurance_provider"].map(prov_rej_map)

df["visit_month"] = df["visit_date"].dt.month
df["visit_quarter"] = df["visit_date"].dt.quarter
df["visit_dayofweek"] = df["visit_date"].dt.dayofweek
df["is_weekend"] = (df["visit_dayofweek"] >= 5).astype(int)
df["high_risk_flag"] = (df["risk_score"] == "High").astype(int)
df["billed_per_hour"] = (df["billed_amount"] / df["length_of_stay_hours"].replace(0, np.nan)).round(2)
df["above_avg_bill"] = (df["billed_amount"] > df["billed_amount"].mean()).astype(int)

# Stratified sampling to guarantee representation across departments, providers, risk, and claim status
# Target 100 samples
sample_groups = []
# Ensure each combination of Department and Risk Score has samples
departments = df["department"].dropna().unique()
risks = df["risk_score"].dropna().unique()
claims = df["claim_status"].dropna().unique()

target_total = 100
# First sample proportionally across Department x Risk Score
per_combo = max(1, target_total // (len(departments) * len(risks)))

sampled_indices = set()
for dept in departments:
    for risk in risks:
        sub = df[(df["department"] == dept) & (df["risk_score"] == risk)]
        if not sub.empty:
            chosen = sub.sample(min(len(sub), per_combo), random_state=42)
            sampled_indices.update(chosen.index.tolist())

# Ensure all insurance providers and claim statuses have strong presence
for prov in df["insurance_provider"].dropna().unique():
    sub = df[df["insurance_provider"] == prov]
    if len(sampled_indices.intersection(sub.index)) < 15:
        avail = sub.index.difference(list(sampled_indices))
        if len(avail) > 0:
            add = np.random.RandomState(42).choice(avail, size=min(len(avail), 8), replace=False)
            sampled_indices.update(add.tolist())

for clm in claims:
    sub = df[df["claim_status"] == clm]
    if len(sampled_indices.intersection(sub.index)) < 20:
        avail = sub.index.difference(list(sampled_indices))
        if len(avail) > 0:
            add = np.random.RandomState(42).choice(avail, size=min(len(avail), 10), replace=False)
            sampled_indices.update(add.tolist())

# Trim or pad to exactly 100
sample_idx_list = list(sampled_indices)
if len(sample_idx_list) > target_total:
    sample_idx_list = sample_idx_list[:target_total]
elif len(sample_idx_list) < target_total:
    remaining = df.index.difference(sample_idx_list)
    needed = target_total - len(sample_idx_list)
    extra = np.random.RandomState(42).choice(remaining, size=needed, replace=False)
    sample_idx_list.extend(extra.tolist())

sample_df = df.loc[sample_idx_list].copy().reset_index(drop=True)

# Load models and compute predictions
model_a_path = find_first_existing(MODEL_A_PATHS)
model_b_path = find_first_existing(MODEL_B_PATHS)

A_NUMERIC = ['age', 'chronic_flag', 'length_of_stay_hours', 'days_since_registration', 'visit_month', 'visit_dayofweek', 'is_weekend']
A_CAT = ['department', 'visit_type', 'gender', 'city', 'insurance_provider']
A_FEAT = A_NUMERIC + A_CAT

B_NUMERIC = ['age', 'chronic_flag', 'length_of_stay_hours', 'billed_amount', 'billed_per_hour', 'above_avg_bill', 'visit_frequency', 'avg_los_per_patient', 'days_since_registration', 'provider_rejection_rate', 'visit_month', 'visit_quarter', 'visit_dayofweek', 'is_weekend', 'high_risk_flag']
B_CAT = ['department', 'visit_type', 'gender', 'city', 'insurance_provider']
B_FEAT = B_NUMERIC + B_CAT

if model_a_path:
    with open(model_a_path, "rb") as f:
        obj_a = pickle.load(f)
    pipe_a = obj_a["model"] if isinstance(obj_a, dict) else obj_a
    le_a = obj_a.get("label_encoder", None) if isinstance(obj_a, dict) else None

    pred_a_raw = pipe_a.predict(sample_df[A_FEAT])
    if le_a is not None:
        sample_df["predicted_risk"] = le_a.inverse_transform(pred_a_raw)
    else:
        sample_df["predicted_risk"] = pred_a_raw
    
    if hasattr(pipe_a, "predict_proba"):
        probs_a = pipe_a.predict_proba(sample_df[A_FEAT])
        classes_a = le_a.classes_ if le_a is not None else getattr(pipe_a, "classes_", [0, 1, 2])
        for i, c in enumerate(classes_a):
            sample_df[f"prob_risk_{str(c).lower()}"] = probs_a[:, i].round(4)
        sample_df["risk_confidence"] = probs_a.max(axis=1).round(4)

if model_b_path:
    with open(model_b_path, "rb") as f:
        pipe_b = pickle.load(f)
    
    pred_b_raw = pipe_b.predict(sample_df[B_FEAT])
    sample_df["predicted_claim_status"] = pred_b_raw

    if hasattr(pipe_b, "predict_proba"):
        probs_b = pipe_b.predict_proba(sample_df[B_FEAT])
        classes_b = pipe_b.classes_
        for i, c in enumerate(classes_b):
            sample_df[f"prob_claim_{str(c).lower()}"] = probs_b[:, i].round(4)
        sample_df["claim_confidence"] = probs_b.max(axis=1).round(4)

# Format dates to string
for col in ["visit_date", "billing_date", "registration_date"]:
    if col in sample_df.columns:
        sample_df[col] = sample_df[col].dt.strftime("%Y-%m-%d")

# Add accuracy match indicators
sample_df["risk_pred_correct"] = (sample_df["risk_score"] == sample_df["predicted_risk"]).astype(bool)
sample_df["claim_pred_correct"] = (sample_df["claim_status"] == sample_df["predicted_claim_status"]).astype(bool)

# Save CSV
csv_out = DATA_DIR / "sample_dataset.csv"
sample_df.to_csv(csv_out, index=False)
print(f"Saved {len(sample_df)} records to {csv_out}")

# Save JSON
json_out = DATA_DIR / "sample_dataset.json"
records = json.loads(sample_df.to_json(orient="records"))
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)
print(f"Saved JSON dataset to {json_out}")

# Summary Stats
print("\nSample Dataset Distribution:")
print("Departments:", sample_df["department"].value_counts().to_dict())
print("Risk Scores (Actual):", sample_df["risk_score"].value_counts().to_dict())
print("Risk Scores (Predicted):", sample_df["predicted_risk"].value_counts().to_dict())
print("Claim Status (Actual):", sample_df["claim_status"].value_counts().to_dict())
print("Claim Status (Predicted):", sample_df["predicted_claim_status"].value_counts().to_dict())
print(f"Model A Accuracy on Sample: {sample_df['risk_pred_correct'].mean():.2%}")
print(f"Model B Accuracy on Sample: {sample_df['claim_pred_correct'].mean():.2%}")
