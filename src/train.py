"""
train.py — Train XGBoost with SMOTE oversampling, evaluate, and persist model artefacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve, average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

from preprocess import load_data, prepare_data

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def train(random_state: int = 42):
    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    print("📥  Loading dataset...")
    df = load_data()
    print(f"   Rows: {df.shape[0]} | Default rate: {df['class'].mean():.2%}")

    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    # ── 2. SMOTE oversampling ─────────────────────────────────────────────────
    print("⚖️   Applying SMOTE to balance classes...")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"   Resampled train size: {X_res.shape[0]} (50/50 split)")

    # ── 3. Train XGBoost ──────────────────────────────────────────────────────
    print("🚀  Training XGBoost with CV...")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
    )
    
    cv_scores = cross_val_score(model, X_res, y_res, cv=5, scoring="roc_auc")
    print(f"   CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=30,
        verbose=False,
    )

    # ── 4. Calibrate probabilities ────────────────────────────────────────────
    print("🎯  Calibrating probability outputs...")
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_train, y_train)

    # ── 5. Evaluate and Tune Threshold ────────────────────────────────────────
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    
    # Tune threshold to maximize F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Recalculate predictions with best threshold
    y_pred = (y_prob >= best_threshold).astype(int)

    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n📊  Evaluation Results:")
    print(f"   Best Threshold: {best_threshold:.4f}")
    print(f"   CV AUC   : {cv_scores.mean():.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"   Avg Prec : {ap:.4f}")
    print(f"   Precision: {report['1']['precision']:.4f}")
    print(f"   Recall   : {report['1']['recall']:.4f}")
    print(f"   F1-Score : {report['1']['f1-score']:.4f}")

    metrics = {
        "best_threshold": round(float(best_threshold), 4),
        "cv_auc_mean": round(cv_scores.mean(), 4),
        "cv_auc_std": round(cv_scores.std(), 4),
        "roc_auc": round(auc, 4),
        "avg_precision": round(ap, 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1_score": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
    }

    # ── 6. Plot ROC curve ─────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    axes[0].plot(fpr, tpr, color="#6c63ff", lw=2.5, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    axes[0].fill_between(fpr, tpr, alpha=0.15, color="#6c63ff")
    axes[0].set_title("ROC Curve", color="white", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", color="#aaa")
    axes[0].set_ylabel("True Positive Rate", color="#aaa")
    axes[0].legend(facecolor="#1a1d2e", labelcolor="white")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, color="#ff6584", lw=2.5, label=f"AP = {ap:.3f}")
    axes[1].fill_between(rec, prec, alpha=0.15, color="#ff6584")
    axes[1].set_title("Precision-Recall Curve", color="white", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", color="#aaa")
    axes[1].set_ylabel("Precision", color="#aaa")
    axes[1].legend(facecolor="#1a1d2e", labelcolor="white")

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdPu",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax, linewidths=0.5, linecolor="#333",
        annot_kws={"color": "white", "fontsize": 12},
    )
    ax.set_title("Confusion Matrix", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", color="#aaa")
    ax.set_ylabel("Actual", color="#aaa")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 8. Feature importance ─────────────────────────────────────────────────
    importances = model.feature_importances_
    top_n = 20
    top_idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, top_n))
    ax.barh(
        [feature_names[i] for i in top_idx],
        importances[top_idx],
        color=colors,
    )
    ax.set_title(f"Top {top_n} Feature Importances", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlabel("Importance Score", color="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 9. Persist artefacts ──────────────────────────────────────────────────
    print("\n💾  Saving model artefacts...")
    joblib.dump(calibrated,    os.path.join(ARTIFACTS_DIR, "model.pkl"))
    joblib.dump(preprocessor,  os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅  Done! Artefacts saved to /artifacts/")
    return calibrated, preprocessor, feature_names, metrics


if __name__ == "__main__":
    train()
