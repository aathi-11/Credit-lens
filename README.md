# Credit Risk Scorer 🏦

An AI-powered loan default prediction system built with XGBoost, SHAP explainability, and a Streamlit UI.

---

## 🚀 Quick Start

### 1. Install dependencies & train the model
```bash
cd "Credit Risk Scorer"
python setup.py
```

### 2. Launch the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Credit Risk Scorer/
├── app.py                  ← Streamlit UI (dark theme, gauges, SHAP)
├── setup.py                ← One-shot install + train
├── requirements.txt
├── src/
│   ├── preprocess.py       ← Data loading + feature engineering
│   ├── train.py            ← XGBoost + SMOTE + calibration + plots
│   ├── explain.py          ← SHAP global & local explainability
│   └── predict.py          ← Map UI inputs → preprocessor → prediction
└── artifacts/              ← Auto-created after training
    ├── model.pkl
    ├── preprocessor.pkl
    ├── feature_names.pkl
    ├── metrics.json
    ├── roc_pr_curve.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## ⚙️ Features

| Feature | Details |
|---------|---------|
| **Dataset** | German Credit (UCI) — 1,000 rows, 20 attrs |
| **Engineered** | 7 domain-specific features added |
| **Imbalance** | SMOTE oversampling |
| **Model** | XGBoost (400 trees) + Isotonic Calibration |
| **Metrics** | ROC-AUC, Precision, Recall, F1, AP |
| **Explainability** | SHAP KernelExplainer (global + local) |
| **UI** | Dark glassmorphism Streamlit app |

---

## 📊 Expected Performance

| Metric | Typical Value |
|--------|--------------|
| ROC-AUC | ~0.78–0.82 |
| Precision (default) | ~0.62–0.70 |
| Recall (default) | ~0.65–0.75 |
| F1-Score | ~0.63–0.70 |

---

## 🔍 How SHAP Works Here

- **Global**: Bar chart showing average feature impact across test set
- **Local**: Per-applicant waterfall chart showing exactly which factors pushed the score up/down
- **Legal compliance**: Mirrors EU AI Act & GDPR Art. 22 explainability requirements

---

## 🧠 Tech Stack

- `xgboost` — gradient boosting
- `imbalanced-learn` — SMOTE oversampling
- `shap` — explainability
- `scikit-learn` — preprocessing pipelines
- `streamlit` — interactive UI
- `plotly` / `matplotlib` / `seaborn` — visualizations
