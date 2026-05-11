"""
explain.py — SHAP-based global and local explainability helpers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def load_artefacts():
    model         = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    preprocessor  = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))
    return model, preprocessor, feature_names


def build_explainer(model, X_background: pd.DataFrame):
    """Build a SHAP KernelExplainer (works for any sklearn-compatible model)."""
    # Use a small background sample for speed
    background = shap.sample(X_background, 50, random_state=42)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer


def global_shap_summary(explainer, X_test: pd.DataFrame, feature_names: list, save: bool = True):
    """Compute and plot SHAP summary (global feature importance)."""
    shap_values = explainer.shap_values(X_test.iloc[:100])  # limit for speed
    # shap_values is list[2] for binary; index 1 = default class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    shap.summary_plot(
        sv, X_test.iloc[:100], feature_names=feature_names,
        plot_type="bar", show=False, color="#6c63ff",
    )
    ax.set_title("SHAP Global Feature Importance", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlabel("Mean |SHAP value|", color="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    if save:
        plt.savefig(os.path.join(ARTIFACTS_DIR, "shap_global.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return sv


def local_shap_explain(explainer, single_row: pd.DataFrame, feature_names: list) -> dict:
    """Return SHAP values for a single prediction (for Streamlit waterfall chart)."""
    sv = explainer.shap_values(single_row)
    if isinstance(sv, list):
        sv = sv[1]  # default class
    shap_dict = {
        name: float(val)
        for name, val in zip(feature_names, sv[0])
    }
    return shap_dict


def waterfall_chart(shap_dict: dict, base_value: float, predicted_prob: float) -> plt.Figure:
    """Render a styled waterfall chart for a single applicant's SHAP values."""
    # Sort by absolute value, take top 12
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names  = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    # Shorten long OHE feature names
    def shorten(name):
        parts = name.split("_")
        return "_".join(parts[:3]) if len(parts) > 3 else name

    names = [shorten(n) for n in names]
    colors = ["#ff6584" if v > 0 else "#6c63ff" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="#222", linewidth=0.5)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right",
            color="white", fontsize=8,
        )

    ax.axvline(0, color="#888", lw=1, linestyle="--")
    ax.set_title(
        f"SHAP Explanation  |  Base: {base_value:.2f}  →  Predicted: {predicted_prob:.2f}",
        color="white", fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("SHAP Value (impact on default probability)", color="#aaa", fontsize=9)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    from preprocess import load_data, prepare_data
    print("Building global SHAP summary...")
    df = load_data()
    _, X_test, _, _, _, _ = prepare_data(df)
    model, preprocessor, feature_names = load_artefacts()
    explainer = build_explainer(model, X_test)
    global_shap_summary(explainer, X_test, feature_names)
    print("SHAP summary saved to artifacts/shap_global.png")
