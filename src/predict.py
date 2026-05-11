"""
predict.py — Build the exact same feature matrix from raw user inputs
             that the trained preprocessor expects, then return a prediction.
"""

import os
import numpy as np
import pandas as pd
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

# ── Raw categorical codes used in the German Credit dataset ──────────────────
CHECKING_STATUS_MAP = {
    "< 0 DM (overdrawn)":     "A11",
    "0 – 200 DM":             "A12",
    ">= 200 DM / salary":     "A13",
    "No checking account":    "A14",
}

CREDIT_HISTORY_MAP = {
    "No credits / all paid duly": "A30",
    "All credits paid duly":      "A31",
    "Existing credits paid duly": "A32",
    "Delay in past":              "A33",
    "Critical account":           "A34",
}

PURPOSE_MAP = {
    "Car (new)":            "A40",
    "Car (used)":           "A41",
    "Furniture / equipment":"A42",
    "Radio / television":   "A43",
    "Domestic appliances":  "A44",
    "Repairs":              "A45",
    "Education":            "A46",
    "Business":             "A48",
    "Other":                "A49",
}

SAVINGS_MAP = {
    "< 100 DM":              "A61",
    "100 – 500 DM":          "A62",
    "500 – 1000 DM":         "A63",
    ">= 1000 DM":            "A64",
    "Unknown / no savings":  "A65",
}

EMPLOYMENT_MAP = {
    "Unemployed":            "A71",
    "< 1 year":              "A72",
    "1 – 4 years":           "A73",
    "4 – 7 years":           "A74",
    ">= 7 years":            "A75",
}

HOUSING_MAP = {
    "Free housing":  "A151",
    "Renting":       "A152",
    "Own property":  "A153",
}

JOB_MAP = {
    "Unemployed / unskilled – non-resident": "A171",
    "Unskilled – resident":                  "A172",
    "Skilled / official":                    "A173",
    "Management / highly qualified":         "A174",
}


def build_input_df(
    checking_status: str,
    duration: int,
    credit_history: str,
    purpose: str,
    credit_amount: int,
    savings_status: str,
    employment: str,
    installment_commitment: int,
    residence_since: int,
    age: int,
    housing: str,
    existing_credits: int,
    job: str,
    num_dependents: int,
    own_telephone: bool,
    foreign_worker: bool,
) -> pd.DataFrame:
    """
    Map human-readable UI inputs to the exact column format expected
    by the preprocessor, including engineered features.
    """
    from preprocess import engineer_features

    row = {
        "checking_status":       CHECKING_STATUS_MAP[checking_status],
        "duration":              duration,
        "credit_history":        CREDIT_HISTORY_MAP[credit_history],
        "purpose":               PURPOSE_MAP[purpose],
        "credit_amount":         credit_amount,
        "savings_status":        SAVINGS_MAP[savings_status],
        "employment":            EMPLOYMENT_MAP[employment],
        "installment_commitment":installment_commitment,
        "personal_status":       "A93",      # defaulted: not collected in UI to avoid bias (gender proxy)
        "other_parties":         "A101",     # defaulted: 'none' to simplify the demo UI
        "residence_since":       residence_since,
        "property_magnitude":    "A121",     # defaulted: 'real estate' (neutral baseline for demo)
        "age":                   age,
        "other_payment_plans":   "A143",     # none
        "housing":               HOUSING_MAP[housing],
        "existing_credits":      existing_credits,
        "job":                   JOB_MAP[job],
        "num_dependents":        num_dependents,
        "own_telephone":         "A191" if not own_telephone else "A192",
        "foreign_worker":        "A201" if foreign_worker else "A202",
        "class":                 0,          # placeholder, dropped during engineering
    }

    df = pd.DataFrame([row])
    df = engineer_features(df)
    df = df.drop(columns=["class"], errors="ignore")
    return df


def predict(input_df: pd.DataFrame):
    """
    Load saved preprocessor + model and return (probability, processed_df).
    """
    preprocessor  = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    model         = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    X_proc = preprocessor.transform(input_df)
    X_df   = pd.DataFrame(X_proc, columns=feature_names)

    prob = model.predict_proba(X_df)[0][1]
    return prob, X_df, feature_names
