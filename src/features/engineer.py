"""
Feature engineering: log transforms, ratios, service bundle aggregates, and interactions.
Called inside preprocess() so both training and inference paths apply the same transformations.
"""
import numpy as np
import pandas as pd


_SECURITY_COLS = [
    "has_online_security", "has_online_backup",
    "has_device_protection", "has_tech_support",
]
_ENTERTAINMENT_COLS = ["has_streaming_tv", "has_streaming_movies"]
_BASIC_COLS = ["has_phone_service", "has_internet_service"]
_ALL_SERVICE_COLS = _SECURITY_COLS + _ENTERTAINMENT_COLS + _BASIC_COLS

# Numerical columns to log-transform (right-skewed distributions)
_LOG_COLS = [
    "annual_income", "totalcharges", "avg_monthly_gb",
    "days_since_last_interaction", "num_complaints",
]


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add engineered features in-place on a copy of df.
    Returns (enriched_df, list_of_new_column_names).
    Must be called AFTER null imputation and BEFORE encoding / scaling.
    """
    df = df.copy()
    new_cols: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. LOG TRANSFORMS for skewed numerical variables                     #
    # ------------------------------------------------------------------ #
    for col in _LOG_COLS:
        if col in df.columns:
            new = f"log_{col}"
            df[new] = np.log1p(df[col].clip(lower=0))
            new_cols.append(new)

    # ------------------------------------------------------------------ #
    # 2. RATIO FEATURES                                                    #
    # ------------------------------------------------------------------ #
    if {"monthlycharges", "num_services"}.issubset(df.columns):
        df["charges_per_service"] = df["monthlycharges"] / (df["num_services"] + 1)
        new_cols.append("charges_per_service")

    if {"monthlycharges", "annual_income"}.issubset(df.columns):
        # Annual charges as fraction of income (affordability pressure)
        df["monthly_income_ratio"] = (df["monthlycharges"] * 12) / (df["annual_income"] + 1)
        new_cols.append("monthly_income_ratio")

    if {"totalcharges", "tenure"}.issubset(df.columns):
        # Normalised average monthly spend corrected by actual tenure
        df["avg_charge_per_tenure"] = df["totalcharges"] / (df["tenure"] + 1)
        new_cols.append("avg_charge_per_tenure")

    if {"num_complaints", "tenure"}.issubset(df.columns):
        df["complaint_rate"] = df["num_complaints"] / (df["tenure"] + 1)
        new_cols.append("complaint_rate")

    if {"num_service_calls", "tenure"}.issubset(df.columns):
        df["service_call_rate"] = df["num_service_calls"] / (df["tenure"] + 1)
        new_cols.append("service_call_rate")

    if {"late_payments", "tenure"}.issubset(df.columns):
        df["late_payment_rate"] = df["late_payments"] / (df["tenure"] + 1)
        new_cols.append("late_payment_rate")

    if {"customer_satisfaction", "num_complaints"}.issubset(df.columns):
        # High satisfaction + zero complaints → strong signal against churn
        df["net_satisfaction"] = df["customer_satisfaction"] / (df["num_complaints"] + 1)
        new_cols.append("net_satisfaction")

    # ------------------------------------------------------------------ #
    # 3. SERVICE BUNDLE AGGREGATES                                         #
    # ------------------------------------------------------------------ #
    present_security = [c for c in _SECURITY_COLS if c in df.columns]
    if present_security:
        df["security_bundle_count"] = df[present_security].sum(axis=1)
        new_cols.append("security_bundle_count")

    present_entertainment = [c for c in _ENTERTAINMENT_COLS if c in df.columns]
    if present_entertainment:
        df["entertainment_bundle_count"] = df[present_entertainment].sum(axis=1)
        new_cols.append("entertainment_bundle_count")

    present_all = [c for c in _ALL_SERVICE_COLS if c in df.columns]
    if present_all:
        df["total_active_services"] = df[present_all].sum(axis=1)
        new_cols.append("total_active_services")

    # ------------------------------------------------------------------ #
    # 4. INTERACTION FEATURES                                              #
    # ------------------------------------------------------------------ #
    if {"senior_citizen", "monthlycharges"}.issubset(df.columns):
        # Seniors paying high monthly charges → elevated churn risk
        df["senior_monthly_charges"] = df["senior_citizen"] * df["monthlycharges"]
        new_cols.append("senior_monthly_charges")

    if {"customer_satisfaction", "tenure"}.issubset(df.columns):
        # Long-tenure + high satisfaction → loyal; new + dissatisfied → risky
        df["satisfaction_x_tenure"] = df["customer_satisfaction"] * df["tenure"]
        new_cols.append("satisfaction_x_tenure")

    # ------------------------------------------------------------------ #
    # 5. TENURE SEGMENT FLAGS                                              #
    # ------------------------------------------------------------------ #
    if "tenure" in df.columns:
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        df["is_loyal_customer"] = (df["tenure"] >= 24).astype(int)
        new_cols += ["is_new_customer", "is_loyal_customer"]

    # ------------------------------------------------------------------ #
    # 6. CONTRACT RISK INTERACTIONS (contract is still string here)        #
    # ------------------------------------------------------------------ #
    _CONTRACT_RISK = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    if "contract" in df.columns:
        risk = df["contract"].map(_CONTRACT_RISK).fillna(1.0)

        if "num_complaints" in df.columns:
            df["contract_x_complaints"] = risk * df["num_complaints"]
            new_cols.append("contract_x_complaints")

        if "customer_satisfaction" in df.columns:
            df["contract_x_satisfaction"] = risk * df["customer_satisfaction"]
            new_cols.append("contract_x_satisfaction")

        if "late_payments" in df.columns:
            df["contract_x_late_payments"] = risk * df["late_payments"]
            new_cols.append("contract_x_late_payments")

        if "complaint_rate" in df.columns:
            df["contract_x_complaint_rate"] = risk * df["complaint_rate"]
            new_cols.append("contract_x_complaint_rate")

    return df, new_cols
