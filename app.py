"""
app.py — Streamlit Credit Risk Scorer UI
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# ── Path fix so src/ imports work ────────────────────────────────────────────
SRC_DIR      = os.path.join(os.path.dirname(__file__), "src")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
sys.path.insert(0, SRC_DIR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Global light background ─── */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 40%, #e2e8f0 100%);
    color: #1e293b;
}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.95) !important;
    border-right: 1px solid rgba(108, 99, 255, 0.2);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* ─── Metric cards ─── */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(108,99,255,0.15);
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff 0%, #a855f7 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2.2rem;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(108,99,255,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.4);
    filter: brightness(1.05);
}

/* ─── Sliders & inputs ─── */
.stSlider > div > div > div > div {
    background: #6c63ff !important;
}
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid rgba(108,99,255,0.3) !important;
    border-radius: 8px !important;
}

/* ─── Headers ─── */
h1 { 
    background: linear-gradient(90deg, #4f46e5, #9333ea, #db2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.4rem;
    margin-bottom: 0 !important;
}
h2, h3 { color: #4338ca; font-weight: 600; }

/* ─── Risk gauge container ─── */
.risk-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.05);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin: 12px 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* ─── Score badge ─── */
.score-badge-low    { background: linear-gradient(135deg,#10b98115,#10b98105); border: 1px solid #10b981; border-radius:12px; padding:18px; }
.score-badge-medium { background: linear-gradient(135deg,#f59e0b15,#f59e0b05); border: 1px solid #f59e0b; border-radius:12px; padding:18px; }
.score-badge-high   { background: linear-gradient(135deg,#ef444415,#ef444405); border: 1px solid #ef4444; border-radius:12px; padding:18px; }

/* ─── Tab styling ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(0,0,0,0.03);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    color: #64748b;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.15) !important;
    color: #4f46e5 !important;
}

/* ─── Divider ─── */
hr { border-color: rgba(108,99,255,0.15) !important; }

/* ─── Info/warning boxes ─── */
.stAlert { border-radius: 10px; }

/* ─── Expander ─── */
details { border: 1px solid rgba(108,99,255,0.2) !important; border-radius: 10px !important; background: #ffffff; }

</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
ARTIFACTS_READY = (
    os.path.exists(os.path.join(ARTIFACTS_DIR, "model.pkl")) and
    os.path.exists(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
)


def load_metrics():
    path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def risk_gauge_chart(prob: float) -> plt.Figure:
    """Draw a semicircular gauge showing default probability."""
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    r = 1.0
    ax.plot(r * np.cos(theta), r * np.sin(theta), lw=18, color="#e2e8f0", solid_capstyle="round")

    # Colored fill
    if prob < 0.35:
        color = "#10b981"
    elif prob < 0.60:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    theta_fill = np.linspace(np.pi, np.pi - prob * np.pi, 200)
    ax.plot(r * np.cos(theta_fill), r * np.sin(theta_fill), lw=18, color=color, solid_capstyle="round")

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.75 * np.cos(angle), 0.75 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1e293b", lw=2, mutation_scale=18))

    # Center dot
    ax.plot(0, 0, "o", color="#1e293b", markersize=8, zorder=5)

    # Labels
    ax.text(-1.1, -0.25, "0%",  color="#64748b", fontsize=9, ha="center", va="center")
    ax.text( 0,    1.22, "50%", color="#64748b", fontsize=9, ha="center", va="center")
    ax.text( 1.1, -0.25, "100%",color="#64748b", fontsize=9, ha="center", va="center")

    # Probability label
    ax.text(0, -0.45, f"{prob:.1%}", color=color, fontsize=22, fontweight="bold", ha="center", va="center")
    ax.text(0, -0.72, "Default Risk", color="#64748b", fontsize=9, ha="center", va="center")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.9, 1.4)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


def waterfall_chart(shap_dict: dict, predicted_prob: float) -> plt.Figure:
    sorted_items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names  = [k for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    def shorten(name):
        parts = name.split("_")
        return "_".join(parts[:3]) if len(parts) > 3 else name

    names  = [shorten(n) for n in names]
    colors = ["#ef4444" if v > 0 else "#6c63ff" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], edgecolor="#e2e8f0", linewidth=0.5, height=0.55)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha="left" if val >= 0 else "right",
            color="#1e293b", fontsize=8.5,
        )

    ax.axvline(0, color="#cbd5e1", lw=1, linestyle="--")
    ax.set_title(
        f"Why this risk score? (predicted default prob = {predicted_prob:.1%})",
        color="#1e293b", fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_xlabel("SHAP value  →  increases risk   |   ←  reduces risk", color="#64748b", fontsize=8.5)
    ax.tick_params(colors="#1e293b", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e8f0")
    plt.tight_layout()
    return fig


# ── Sidebar: Train model ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Credit Risk Scorer")
    st.markdown("*Powered by XGBoost + SHAP*")
    st.markdown("---")

    if not ARTIFACTS_READY:
        st.warning("⚠️ Model not trained yet.")
        st.markdown("Click below to download the German Credit dataset and train the model (~30 sec).")
        if st.button("🚀 Train Model Now", key="train_btn"):
            with st.spinner("Downloading data & training model…"):
                try:
                    from train import train
                    calibrated, preprocessor, feature_names, metrics = train()
                    st.success("✅ Model trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
    else:
        st.success("✅ Model ready")
        metrics = load_metrics()
        if metrics:
            st.markdown("### 📊 Model Performance")
            if "cv_auc_mean" in metrics:
                st.metric("CV AUC", f"{metrics['cv_auc_mean']:.4f} ± {metrics.get('cv_auc_std', 0):.4f}")
            st.metric("ROC-AUC",  f"{metrics.get('roc_auc', 0):.4f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            st.metric("Recall",    f"{metrics.get('recall', 0):.4f}")
            st.metric("F1-Score",  f"{metrics.get('f1_score', 0):.4f}")

    st.markdown("---")
    st.markdown("""
    **About this tool**

    This scorer mimics how banks assess credit applications using:
    - ✅ 20+ engineered features
    - ✅ XGBoost with SMOTE balancing
    - ✅ Calibrated probabilities
    - ✅ SHAP explainability
    - ✅ Decision transparency
    """)


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown("# 🏦 Credit Risk Scorer")
st.markdown("*AI-powered loan default prediction with full explainability*")
st.markdown("---")

if not ARTIFACTS_READY:
    st.info("👈 Please train the model first using the sidebar button.")
    st.stop()

# Load artefacts
from predict import (
    build_input_df, predict,
    CHECKING_STATUS_MAP, CREDIT_HISTORY_MAP, PURPOSE_MAP,
    SAVINGS_MAP, EMPLOYMENT_MAP, HOUSING_MAP, JOB_MAP,
)

tabs = st.tabs(["🔍 Risk Assessment", "📊 Model Analytics", "📚 About", "🗂️ Batch Scoring"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("### 👤 Applicant Profile")
    st.markdown("Fill in the applicant's details below and click **Assess Risk**.")

    with st.form("applicant_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💳 Financial Info**")
            credit_amount = st.number_input(
                "Credit Amount (DM)", min_value=500, max_value=20000, value=3000, step=100,
                help="Loan amount requested in Deutschmarks"
            )
            duration = st.slider("Loan Duration (months)", 6, 72, 24)
            installment_commitment = st.slider(
                "Installment Rate (% of income)", 1, 4, 2,
                help="Monthly instalment as % of disposable income"
            )
            checking_status = st.selectbox(
                "Checking Account Status", list(CHECKING_STATUS_MAP.keys())
            )
            savings_status = st.selectbox(
                "Savings Account", list(SAVINGS_MAP.keys())
            )

        with col2:
            st.markdown("**👤 Personal Details**")
            age = st.slider("Age", 18, 75, 35)
            employment = st.selectbox(
                "Employment Duration", list(EMPLOYMENT_MAP.keys())
            )
            housing = st.selectbox(
                "Housing Situation", list(HOUSING_MAP.keys())
            )
            job = st.selectbox(
                "Job Category", list(JOB_MAP.keys())
            )
            num_dependents = st.slider("Number of Dependents", 1, 2, 1)
            own_telephone = st.checkbox("Has Telephone")
            foreign_worker = st.checkbox("Foreign Worker", value=True)

        with col3:
            st.markdown("**📋 Credit History**")
            credit_history = st.selectbox(
                "Credit History", list(CREDIT_HISTORY_MAP.keys())
            )
            purpose = st.selectbox(
                "Loan Purpose", list(PURPOSE_MAP.keys())
            )
            existing_credits = st.slider(
                "Existing Credits at Bank", 1, 4, 1,
                help="Number of existing credit lines at this bank"
            )
            residence_since = st.slider("Years at Current Residence", 1, 4, 2)
            
        st.markdown("---")
        st.markdown("**⚙️ Risk Settings**")
        default_thresh = 0.5
        metrics = load_metrics()
        if metrics and "best_threshold" in metrics:
            default_thresh = float(metrics["best_threshold"])
        threshold = st.slider("Approval Risk Threshold", 0.1, 0.9, default_thresh, 0.05, help="Maximum default probability allowed to approve.")

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        with st.spinner("Running credit risk model…"):
            input_df = build_input_df(
                checking_status=checking_status,
                duration=duration,
                credit_history=credit_history,
                purpose=purpose,
                credit_amount=credit_amount,
                savings_status=savings_status,
                employment=employment,
                installment_commitment=installment_commitment,
                residence_since=residence_since,
                age=age,
                housing=housing,
                existing_credits=existing_credits,
                job=job,
                num_dependents=num_dependents,
                own_telephone=own_telephone,
                foreign_worker=foreign_worker,
            )
            prob, X_proc, feature_names = predict(input_df)

        st.markdown("---")
        st.markdown("### 📊 Risk Assessment Result")

        # ── Gauge + verdict ───────────────────────────────────────────────────
        col_gauge, col_verdict = st.columns([1, 1])

        with col_gauge:
            fig_gauge = risk_gauge_chart(prob)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

        with col_verdict:
            credit_score = int(850 - prob * 550)  # map to 300-850 range

            decision = "APPROVE" if prob < threshold else "REJECT"

            if decision == "APPROVE":
                verdict_html = f"""
                <div class="score-badge-low" style="text-align:center">
                    <div style="font-size:3rem">✅</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#10b981;margin:8px 0">APPROVE</div>
                    <div style="color:#94a3b8;font-size:0.9rem">Risk ({prob:.1%}) is below threshold ({threshold:.2f})</div>
                    <div style="margin-top:14px;font-size:0.85rem;color:#64748b">Estimated Credit Score</div>
                    <div style="font-size:2rem;font-weight:700;color:#10b981">{credit_score}</div>
                </div>
                """
            else:
                verdict_html = f"""
                <div class="score-badge-high" style="text-align:center">
                    <div style="font-size:3rem">❌</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#ef4444;margin:8px 0">REJECT</div>
                    <div style="color:#94a3b8;font-size:0.9rem">Risk ({prob:.1%}) exceeds threshold ({threshold:.2f})</div>
                    <div style="margin-top:14px;font-size:0.85rem;color:#64748b">Estimated Credit Score</div>
                    <div style="font-size:2rem;font-weight:700;color:#ef4444">{credit_score}</div>
                </div>
                """
            st.markdown(verdict_html, unsafe_allow_html=True)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Default Probability", f"{prob:.1%}")
            m2.metric("Credit Score (est.)", credit_score)
            m3.metric("Risk Band", "Low" if prob < 0.35 else "Medium" if prob < 0.60 else "High")

        # ── SHAP Explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Why this decision? (SHAP Explanation)")
        st.caption("Red bars **increase** default risk | Blue bars **decrease** risk")

        with st.spinner("Computing SHAP values…"):
            try:
                import shap
                preprocessor = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
                model        = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))

                # Background data for explainer (cached in session)
                if "shap_background" not in st.session_state:
                    from preprocess import load_data, prepare_data
                    df_bg = load_data()
                    _, X_test_bg, _, _, _, _ = prepare_data(df_bg)
                    bg_sample = shap.sample(X_test_bg, 40, random_state=42)
                    st.session_state["shap_background"] = bg_sample

                bg = st.session_state["shap_background"]
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                sv = explainer.shap_values(X_proc, nsamples=80)
                sv_default = sv[1] if isinstance(sv, list) else sv

                shap_dict = {
                    name: float(val)
                    for name, val in zip(feature_names, sv_default[0])
                }

                fig_wf = waterfall_chart(shap_dict, prob)
                st.pyplot(fig_wf, use_container_width=True)
                plt.close()

                # Top factors table
                with st.expander("📋 Full SHAP factor table"):
                    shap_df = pd.DataFrame(
                        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True),
                        columns=["Feature", "SHAP Impact"]
                    )
                    shap_df["Direction"] = shap_df["SHAP Impact"].apply(
                        lambda v: "🔴 Increases Risk" if v > 0 else "🔵 Reduces Risk"
                    )
                    shap_df["SHAP Impact"] = shap_df["SHAP Impact"].map("{:+.4f}".format)
                    st.dataframe(shap_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.warning(f"SHAP explanation not available: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("### 📊 Model Performance Analytics")

    metrics = load_metrics()
    if metrics:
        cols = st.columns(5)
        labels = ["ROC-AUC", "Avg Precision", "Precision", "Recall", "F1-Score"]
        keys   = ["roc_auc", "avg_precision", "precision", "recall", "f1_score"]
        for col, lbl, key in zip(cols, labels, keys):
            col.metric(lbl, f"{metrics.get(key, 0):.4f}")

    st.markdown("---")

    img_col1, img_col2 = st.columns(2)
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_pr_curve.png")
    cm_path  = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    fi_path  = os.path.join(ARTIFACTS_DIR, "feature_importance.png")

    if os.path.exists(roc_path):
        with img_col1:
            st.markdown("**ROC & Precision-Recall Curves**")
            st.image(roc_path, use_column_width=True)

    if os.path.exists(cm_path):
        with img_col2:
            st.markdown("**Confusion Matrix**")
            st.image(cm_path, use_column_width=True)

    if os.path.exists(fi_path):
        st.markdown("---")
        st.markdown("**Top Feature Importances (XGBoost)**")
        st.image(fi_path, use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("### 📚 About This Project")
    st.markdown("""
    **Credit Risk Scorer** is a production-grade ML system that predicts the probability
    of loan default using the [German Credit Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)).

    #### 🏗️ Architecture

    | Layer | Technology |
    |-------|-----------|
    | Data  | German Credit Dataset (1000 rows, 20 features) |
    | Preprocessing | Scikit-learn Pipeline (imputation + scaling + OHE) |
    | Imbalance | SMOTE oversampling |
    | Model | XGBoost Classifier (400 trees) |
    | Calibration | Isotonic Regression (for accurate probabilities) |
    | Explainability | SHAP KernelExplainer |
    | UI | Streamlit |

    #### ⚙️ Engineered Features
    Beyond the raw 20 attributes, we create:
    - `loan_to_income_proxy` — monthly credit burden
    - `payment_to_duration` — installment pressure
    - `age_credit_ratio` — creditworthiness maturity
    - `is_high_risk_purpose` — purpose risk flag
    - `savings_score` — ordinal savings stability
    - `employment_score` — ordinal job stability
    - `checking_score` — account health indicator

    #### 🔍 Explainability
    Banks are **legally required** to explain credit decisions (EU AI Act, GDPR Art. 22).
    This tool uses SHAP (SHapley Additive exPlanations) to show:
    - **Global**: which features matter most across all applicants
    - **Local**: exactly why *this* applicant got *this* score

    #### 📊 Dataset
    - **Source**: UCI Machine Learning Repository
    - **Size**: 1,000 applicants
    - **Default rate**: ~30%
    - **Features**: 20 raw + 7 engineered = 27 total
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Batch Scoring
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("### 🗂️ Batch Scoring")
    st.markdown("Upload a CSV file containing multiple applicants to score them all at once. Ensure the CSV has columns matching the input form.")
    
    uploaded_file = st.file_uploader("Upload Applicants CSV", type="csv")
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head(), use_container_width=True)
        
        if st.button("Score Batch"):
            with st.spinner("Scoring batch..."):
                try:
                    # In a real app we would map columns dynamically, 
                    # but here we rely on the predict module directly or expect processed data
                    preprocessor  = joblib.load(os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))
                    model         = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
                    
                    # For demo purposes, we will try to pass it to preprocessor 
                    # (assuming the uploaded CSV is already in the right schema, e.g., the test set)
                    # We will catch errors and warn the user.
                    from preprocess import engineer_features
                    try:
                        proc_df = engineer_features(batch_df)
                        if "class" in proc_df.columns:
                            proc_df = proc_df.drop(columns=["class"])
                        X_proc = preprocessor.transform(proc_df)
                        probs = model.predict_proba(X_proc)[:, 1]
                        
                        results_df = batch_df.copy()
                        results_df["Default Probability"] = probs
                        
                        default_thresh = 0.5
                        metrics = load_metrics()
                        if metrics and "best_threshold" in metrics:
                            default_thresh = float(metrics["best_threshold"])
                        
                        results_df["Decision"] = ["REJECT" if p >= default_thresh else "APPROVE" for p in probs]
                        
                        st.success(f"Successfully scored {len(results_df)} applicants!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results as CSV",
                            csv,
                            "batch_scoring_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    except Exception as e:
                        st.error(f"Error processing features: {e}. Please ensure the CSV has the exact raw schema as the German Credit dataset.")
                except Exception as e:
                    st.error(f"Scoring failed: {e}")
