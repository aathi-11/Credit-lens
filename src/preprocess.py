"""
preprocess.py — Load and engineer features from the German Credit Dataset (UCI).
The dataset is embedded here as raw data so no external download is needed.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# ── German Credit Dataset column names ───────────────────────────────────────
COLUMN_NAMES = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class"
]

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_parties", "property_magnitude",
    "other_payment_plans", "housing", "job", "own_telephone", "foreign_worker"
]

NUMERICAL_FEATURES = [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
    # engineered
    "loan_to_income_proxy", "payment_to_duration", "age_credit_ratio"
]


def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Download the German Credit dataset and assign column names."""
    df = pd.read_csv(url, sep=" ", header=None, names=COLUMN_NAMES)
    # Target: 1 = bad credit, 2 = good credit → remap to 1 = default, 0 = no default
    df["class"] = df["class"].map({2: 0, 1: 1})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific engineered features."""
    df = df.copy()

    # Proxy for loan-to-income (dataset has no income, use credit_amount / duration as monthly burden)
    df["loan_to_income_proxy"] = df["credit_amount"] / (df["duration"] + 1)

    # Payment burden: installment commitment per month of credit
    df["payment_to_duration"] = df["installment_commitment"] / (df["duration"] + 1)

    # Age relative to credit history length proxy
    df["age_credit_ratio"] = df["age"] / (df["existing_credits"] + 1)

    # High-risk purpose flag
    high_risk_purposes = {"A43", "A44", "A45", "A49"}  # furniture, appliances, repairs, others
    df["is_high_risk_purpose"] = df["purpose"].isin(high_risk_purposes).astype(int)

    # Savings stability score (ordinal encode savings_status)
    savings_map = {
        "A61": 1,  # < 100 DM
        "A62": 2,  # 100–500 DM
        "A63": 3,  # 500–1000 DM
        "A64": 4,  # >= 1000 DM
        "A65": 5,  # no savings / unknown
    }
    df["savings_score"] = df["savings_status"].map(savings_map).fillna(1)

    # Employment stability score
    emp_map = {
        "A71": 0,  # unemployed
        "A72": 1,  # < 1 year
        "A73": 2,  # 1–4 years
        "A74": 3,  # 4–7 years
        "A75": 4,  # >= 7 years
    }
    df["employment_score"] = df["employment"].map(emp_map).fillna(0)

    # Checking account stress (no/overdrawn account = high risk)
    checking_map = {
        "A11": 0,  # < 0 DM (overdrawn)
        "A12": 1,  # 0–200 DM
        "A13": 2,  # >= 200 DM
        "A14": 3,  # no checking account
    }
    df["checking_score"] = df["checking_status"].map(checking_map).fillna(0)

    return df


def build_preprocessor():
    """Build a ColumnTransformer for numerical + categorical features."""
    num_features = NUMERICAL_FEATURES + ["is_high_risk_purpose", "savings_score", "employment_score", "checking_score"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split, preprocess and return train/test sets + preprocessor."""
    df = engineer_features(df)
    target = df["class"]
    features = df.drop(columns=["class"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state, stratify=target
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Build feature names for SHAP
    num_features = NUMERICAL_FEATURES + ["is_high_risk_purpose", "savings_score", "employment_score", "checking_score"]
    cat_names = preprocessor.named_transformers_["cat"]["ohe"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = num_features + list(cat_names)

    X_train_df = pd.DataFrame(X_train_proc, columns=all_feature_names)
    X_test_df  = pd.DataFrame(X_test_proc,  columns=all_feature_names)

    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True), preprocessor, all_feature_names


if __name__ == "__main__":
    print("Loading German Credit Dataset...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Default rate: {df['class'].mean():.2%}")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print("Preprocessing complete.")
