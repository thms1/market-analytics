"""
Telco Data Science Pipeline — Streamlit Web Application
==========================================================
End-to-end data science roadmap with a step-by-step UI:

  Step 1  Data Ingestion          Upload CSV / use demo data
  Step 2  Exploratory Analysis    Stats, distributions, correlations
  Step 3  Data Cleaning           Nulls, duplicates, type coercion
  Step 4  Feature Engineering     Encoding, scaling, derived KPIs
  Step 5  Model Training          Six classifiers compared side-by-side
  Step 6  Model Evaluation        Accuracy, ROC-AUC, F1, confusion matrix
  Step 7  Bias Detection          Learning curves – overfit / underfit
  Step 8  Recommendations         Product & campaign segmentation
  Step 9  Data Story              Executive + technical PowerPoint exports

Run:
    py -3.12 -m streamlit run Data_Science_Pipeline.py
"""

from __future__ import annotations

import io
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco DS Pipeline",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .step-header{font-size:1.6rem;font-weight:700;color:#1f4e79;
        border-bottom:3px solid #2196f3;padding-bottom:10px;margin-bottom:20px;}
    .kpi-card{background:linear-gradient(135deg,#1565c0,#42a5f5);
        border-radius:14px;padding:18px 12px;color:#fff;text-align:center;margin:4px 0;}
    .kpi-value{font-size:2.2rem;font-weight:800;line-height:1.1;}
    .kpi-label{font-size:.82rem;opacity:.88;margin-top:4px;}
    .kpi-card.green{background:linear-gradient(135deg,#2e7d32,#66bb6a);}
    .kpi-card.orange{background:linear-gradient(135deg,#e65100,#ffa726);}
    .kpi-card.purple{background:linear-gradient(135deg,#4a148c,#ab47bc);}
    .insight-box{background:#e3f2fd;border-left:5px solid #1976d2;
        padding:12px 16px;border-radius:4px;margin:8px 0;font-size:.95rem;}
    .warn-box{background:#fff8e1;border-left:5px solid #f57c00;
        padding:12px 16px;border-radius:4px;margin:8px 0;font-size:.95rem;}
    .ok-box{background:#e8f5e9;border-left:5px solid #388e3c;
        padding:12px 16px;border-radius:4px;margin:8px 0;font-size:.95rem;}
    .section-divider{border:none;border-top:2px dashed #90caf9;margin:20px 0;}
    .reco-card{border-radius:10px;padding:14px;margin:6px 0;color:#fff;font-size:.9rem;}
    .reco-red{background:linear-gradient(90deg,#c62828,#ef5350);}
    .reco-amber{background:linear-gradient(90deg,#e65100,#ff7043);}
    .reco-blue{background:linear-gradient(90deg,#1565c0,#42a5f5);}
    .reco-green{background:linear-gradient(90deg,#2e7d32,#66bb6a);}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
STEPS: tuple[tuple[int, str, str], ...] = (
    (1, "📥", "Data Ingestion"),
    (2, "🔍", "Exploratory Analysis"),
    (3, "🧹", "Data Cleaning"),
    (4, "⚙️", "Feature Engineering"),
    (5, "🤖", "Model Training"),
    (6, "📊", "Model Evaluation"),
    (7, "📉", "Bias Detection"),
    (8, "💡", "Recommendations"),
    (9, "🎯", "Data Story"),
)

FIELD_DICT: tuple[tuple[str, str, str, str], ...] = (
    ("subscriber_id", "Identity", "Text", "Unique subscriber key  e.g. SUB00001"),
    ("account_number", "Identity", "Text", "Billing account reference"),
    ("msisdn", "Identity", "Text", "Mobile number in E.164 format (2760xxxxxxx)"),
    ("id_number", "Identity", "Text", "South African ID number (13 digits)"),
    ("plan", "Subscription", "Text", "Price plan name"),
    ("contract_type", "Subscription", "Text", "Month-to-Month / One Year / Two Year"),
    ("network_type", "Subscription", "Text", "2G / 3G / 4G LTE / 5G / NB-IoT"),
    ("service_status", "Subscription", "Text", "Active / Suspended / Terminated / Porting Out"),
    ("acquisition_channel", "Subscription", "Text", "How subscriber was acquired"),
    ("payment_method", "Subscription", "Text", "Credit Card / Debit Order / EFT / Voucher / Mobile Money"),
    ("region", "Subscription", "Text", "Province"),
    ("device_type", "Subscription", "Text", "Smartphone / Feature Phone / IoT Device etc."),
    ("node_id", "Network", "Text", "Base transceiver station ID"),
    ("tenure_months", "Tenure", "Integer", "Months since activation"),
    ("activation_date", "Tenure", "Date", "Service activation date  YYYY-MM-DD"),
    ("created_date", "Dates", "DateTime", "Ticket / transaction created  ISO 8601"),
    ("closed_date", "Dates", "DateTime", "Ticket / transaction closed  ISO 8601"),
    ("last_recharge_days_ago", "Tenure", "Integer", "Days since last recharge event"),
    ("sla_hours", "SLA", "Integer", "SLA target in hours  4/8/24/48"),
    ("monthly_charges", "Billing", "Float", "Monthly invoice amount (ZAR) — may contain '-' tokens"),
    ("data_volume_mb", "Usage", "Float", "Total data consumed in MB — ~4% nulls injected"),
    ("data_overage_mb", "Usage", "Float", "Data consumed above plan bundle"),
    ("call_minutes", "Usage", "Float", "Voice call minutes — ~4% nulls injected"),
    ("sms_count", "Usage", "Integer", "SMS messages sent"),
    ("intl_call_minutes", "Usage", "Float", "International call minutes (0 if not roaming)"),
    ("roaming_flag", "Usage", "Binary", "1 = subscriber roamed in the period"),
    ("avg_recharge_amount", "Usage", "Float", "Average top-up amount (0 for postpaid)"),
    ("latency_ms", "Network KPI", "Float", "Average round-trip latency (ms)"),
    ("signal_strength_dbm", "Network KPI", "Float", "Average signal strength (dBm, -110 to -50)"),
    ("complaint_count", "CX", "Integer", "Complaints logged in period"),
    ("support_calls", "CX", "Integer", "Calls to support centre"),
    ("nps_score", "CX", "Integer", "Net Promoter Score  0-10"),
    ("csat_score", "CX", "Float", "Customer Satisfaction score  1.0-5.0"),
    ("app_logins_monthly", "CX", "Integer", "Self-service app logins per month"),
    ("refund_amount", "Billing", "Float", "Refunds issued (ZAR)"),
    ("payment_failures", "Billing", "Integer", "Failed payment attempts"),
    ("days_overdue", "Billing", "Integer", "Days payment is overdue"),
    ("bill_shock_flag", "Billing", "Binary", "1 = bill > R800 or data overage > 1 GB"),
    ("paperless_billing", "Billing", "Binary", "1 = enrolled in e-billing"),
    ("plan_upgrades", "Loyalty", "Integer", "Number of plan upgrades"),
    ("reactivation_count", "Loyalty", "Integer", "Times reactivated after suspension"),
    ("churn", "TARGET", "Binary", "1 = churned in period  (predict this)"),
)

PIPELINE_CHECKLIST: tuple[tuple[str, str, str], ...] = (
    ("Step 1 Ingestion", "All 42 columns", "CSV upload / demo data load"),
    ("Step 2 EDA", "All numeric columns", "Distributions, correlation heatmap, class balance"),
    ("Step 2 EDA", "plan, region, device_type", "Categorical bar + churn rate charts"),
    ("Step 2 EDA", "data_volume_mb, latency_ms", "Missing value map (4% injected)"),
    ("Step 3 Cleaning", "monthly_charges", "Auto-coerce '-' tokens to numeric"),
    ("Step 3 Cleaning", "N/A, null, None tokens", "Null-token normalisation (payment_method, contract_type, service_status)"),
    ("Step 3 Cleaning", "Blank strings", "Whitespace strip + blank -> null (plan, region, device_type, channel)"),
    ("Step 3 Cleaning", "10 duplicate rows", "Duplicate row removal"),
    ("Step 3 Cleaning", "6 numeric cols", "Median/mean imputation of NaN values"),
    ("Step 4 Engineering", "monthly_charges, data_volume_mb", "Derived: charge_per_mb"),
    ("Step 4 Engineering", "complaint_count, tenure_months", "Derived: complaint_rate"),
    ("Step 4 Engineering", "call_minutes, sms_count, data_volume_mb", "Derived: engagement_score"),
    ("Step 4 Engineering", "tenure_months", "Derived: tenure_bucket (binned)"),
    ("Step 4 Engineering", "created_date, closed_date, sla_hours", "SLA status + resolution_hours"),
    ("Step 4 Engineering", "plan, region, device_type etc.", "One-hot encoding of categoricals"),
    ("Step 5 Training", "churn (target)", "Six models: LR, RF, GB, SVM, KNN, AdaBoost"),
    ("Step 5 Training", "All numeric features", "Feature importance (Random Forest)"),
    ("Step 6 Evaluation", "churn", "Confusion matrix, ROC-AUC, F1, precision, recall"),
    ("Step 6 Evaluation", "churn_probability", "Probability distribution histogram"),
    ("Step 7 Bias Detection", "All features", "Learning curves: overfit / underfit / good fit"),
    ("Step 7 Bias Detection", "Train vs Test", "Train/Test AUC gap per model"),
    ("Step 8 Recommendations", "churn_probability", "7 risk segments with product + campaign mapping"),
    ("Step 9 Data Story", "All stages", "Executive + technical PowerPoint decks + JSON export"),
)

_STATE_DEFAULTS: dict = {
    "current_step": 1,
    "raw_df": None,
    "cleaned_df": None,
    "featured_df": None,
    "target_col": "churn",
    "feature_cols": [],
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "models": {},
    "model_results": {},
    "feature_importance": None,
    "recommendations_df": None,
    "steps_done": set(),
    "eda_insights": [],
    "clean_log": [],
    "dataset_name": "No dataset loaded",
    "demo": False,
}

SAMPLE_CSV_PATH = Path(__file__).resolve().parent / "telco_sample_data.csv"

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
ss = st.session_state
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in ss:
        ss[_k] = _v if _k != "steps_done" else set()

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _nav_sidebar() -> None:
    st.sidebar.markdown("## 📡 Telco DS Pipeline")
    st.sidebar.caption("End-to-end Data Science Roadmap")

    done = ss.steps_done
    for num, icon, label in STEPS:
        unlocked = num == 1 or (num - 1) in done
        marker = "✅" if num in done else ("🔵" if num == ss.current_step else "○")
        if st.sidebar.button(
            f"{marker} {icon} {label}",
            key=f"nav_{num}",
            use_container_width=True,
            disabled=not unlocked,
        ):
            ss.current_step = num

    pct = int(len(done) / len(STEPS) * 100)
    st.sidebar.progress(pct / 100, text=f"Progress: {pct}%")

    if ss.raw_df is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"**{ss.dataset_name}**  \n"
            f"{ss.raw_df.shape[0]:,} rows × {ss.raw_df.shape[1]} cols  \n"
            f"🎯 Target: `{ss.target_col}`"
        )

    if st.sidebar.button("🔄 Reset Pipeline", use_container_width=True):
        for k, v in _STATE_DEFAULTS.items():
            ss[k] = v if k != "steps_done" else set()
        st.rerun()


def _step_header(num: int, title: str, subtitle: str = "") -> None:
    icon = STEPS[num - 1][1]
    st.markdown(f'<div class="step-header">Step {num} {icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(subtitle)


def _complete_step(num: int) -> None:
    ss.steps_done.add(num)
    ss.current_step = min(num + 1, len(STEPS))


def _kpi(label: str, value, color: str = "") -> None:
    cls = f"kpi-card {color}".strip()
    st.markdown(
        f'<div class="{cls}"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _insight(msg: str) -> None:
    st.markdown(f'<div class="insight-box">💡 {msg}</div>', unsafe_allow_html=True)


def _warn(msg: str) -> None:
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def _ok(msg: str) -> None:
    st.markdown(f'<div class="ok-box">✔ {msg}</div>', unsafe_allow_html=True)


# Columns that should never be one-hot / label encoded as model inputs
_SKIP_ENCODE_COLS = {
    "subscriber_id", "msisdn", "id_number", "account_number",
    "activation_date", "created_date", "closed_date",
}
_MAX_OHE_CARDINALITY = 30


def _prepare_binary_target(target: pd.Series, col_name: str) -> pd.Series | None:
    """Coerce target to binary 0/1; return None if not usable for classification."""
    t = target.copy()
    if t.dtype == object:
        t = t.astype(str).str.strip()
        numeric = pd.to_numeric(t, errors="coerce")
        if numeric.notna().mean() > 0.9:
            t = numeric
        else:
            le = LabelEncoder()
            t = pd.Series(le.fit_transform(t), index=target.index, name=col_name)

    t = pd.to_numeric(t, errors="coerce")
    if t.isnull().any():
        _warn(f"Dropping {int(t.isnull().sum())} rows with missing target values.")

    unique = sorted(t.dropna().unique())
    if len(unique) == 1:
        st.error(f"Target `{col_name}` has only one class ({unique[0]}). Cannot train a classifier.")
        return None
    if len(unique) > 2:
        st.error(
            f"Target `{col_name}` has {len(unique)} classes ({unique[:8]}…). "
            "Select a binary churn column (0/1 or Yes/No)."
        )
        return None

    mapping = {unique[0]: 0, unique[1]: 1}
    return t.map(mapping).astype(int).rename(col_name)


def _safe_train_test_split(feat_df: pd.DataFrame, target: pd.Series):
    """Split with stratification when every class has at least 2 samples."""
    split_kwargs: dict = {"test_size": 0.2, "random_state": 42}
    counts = target.value_counts()
    if len(counts) >= 2 and counts.min() >= 2:
        split_kwargs["stratify"] = target
    else:
        rare = counts[counts < 2]
        _warn(
            "Stratified split skipped — some target classes have fewer than 2 samples "
            f"({', '.join(f'{k}: {v}' for k, v in rare.items())}). "
            "Using random split instead."
        )
    return train_test_split(feat_df, target, **split_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
def _demo_data(n: int = 2000) -> pd.DataFrame:
    """
    Full-featured synthetic telco dataset with 42 columns covering:
    Identity, Subscription, Usage, Network KPIs, Customer Experience,
    Billing, Loyalty — plus injected nulls, bad tokens & duplicates
    to exercise every pipeline step.
    """
    rng = np.random.default_rng(42)
    plans = [
        "Prepaid Basic", "Prepaid Smart", "Postpaid 199", "Postpaid 499",
        "Postpaid 999", "IoT SIM", "Business Pro",
    ]
    regions = [
        "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Limpopo",
        "Mpumalanga", "North West", "Free State", "Northern Cape",
    ]
    devices = ["Smartphone", "Feature Phone", "IoT Device", "Tablet", "Router", "Wearable"]
    networks = ["4G LTE", "3G", "5G", "2G", "NB-IoT"]
    channels = ["Direct Sales", "Online Portal", "Retail Store", "Agent", "Referral"]
    contracts = ["Month-to-Month", "One Year", "Two Year"]
    payments = ["Credit Card", "Debit Order", "EFT", "Voucher", "Mobile Money"]
    statuses = ["Active", "Suspended", "Terminated", "Porting Out"]
    nodes = [
        "BTS_JHB_001", "BTS_CPT_007", "BTS_DBN_003",
        "BTS_PTA_012", "BTS_EL_005", "BTS_NM_009",
    ]

    idx = np.arange(n)
    base_date = datetime(2024, 1, 1)

    tenure = rng.integers(1, 72, n)
    activation = [base_date + timedelta(days=int(rng.integers(0, 900))) for _ in idx]
    created = [a + timedelta(hours=int(rng.integers(1, 500))) for a in activation]
    closed = [c + timedelta(hours=int(rng.integers(1, 96))) for c in created]

    monthly_charges = rng.uniform(199, 1200, n).round(2)
    data_volume = rng.uniform(50, 8000, n).round(1)
    call_minutes = rng.uniform(10, 500, n).round(1)
    complaint_count = rng.integers(0, 6, n)
    churn = (
        (complaint_count >= 3) | (monthly_charges > 900) | (data_volume < 200)
    ).astype(int)
    churn = np.where(rng.random(n) < 0.12, 1 - churn, churn)

    df = pd.DataFrame({
        "subscriber_id": [f"SUB{i:05d}" for i in idx],
        "account_number": [f"ACC{rng.integers(100000, 999999)}" for _ in idx],
        "msisdn": [f"2760{rng.integers(1000000, 9999999)}" for _ in idx],
        "id_number": [f"{rng.integers(1000000000000, 9999999999999)}" for _ in idx],
        "plan": rng.choice(plans, n),
        "contract_type": rng.choice(contracts, n),
        "network_type": rng.choice(networks, n),
        "service_status": rng.choice(statuses, n, p=[0.82, 0.08, 0.07, 0.03]),
        "acquisition_channel": rng.choice(channels, n),
        "payment_method": rng.choice(payments, n),
        "region": rng.choice(regions, n),
        "device_type": rng.choice(devices, n),
        "node_id": rng.choice(nodes, n),
        "tenure_months": tenure,
        "activation_date": [d.strftime("%Y-%m-%d") for d in activation],
        "created_date": [d.strftime("%Y-%m-%dT%H:%M:%S") for d in created],
        "closed_date": [d.strftime("%Y-%m-%dT%H:%M:%S") for d in closed],
        "last_recharge_days_ago": rng.integers(1, 90, n),
        "sla_hours": rng.choice([4, 8, 24, 48], n),
        "monthly_charges": monthly_charges,
        "data_volume_mb": data_volume,
        "data_overage_mb": rng.uniform(0, 500, n).round(1),
        "call_minutes": call_minutes,
        "sms_count": rng.integers(0, 80, n),
        "intl_call_minutes": np.where(rng.random(n) < 0.15, rng.uniform(0, 200, n).round(2), 0),
        "roaming_flag": rng.integers(0, 2, n),
        "avg_recharge_amount": np.where(
            rng.random(n) < 0.4, rng.uniform(20, 200, n).round(2), 0
        ),
        "latency_ms": rng.uniform(80, 350, n).round(1),
        "signal_strength_dbm": rng.uniform(-110, -50, n).round(1),
        "complaint_count": complaint_count,
        "support_calls": rng.integers(0, 5, n),
        "nps_score": rng.integers(0, 11, n),
        "csat_score": rng.uniform(1, 5, n).round(1),
        "app_logins_monthly": rng.integers(0, 30, n),
        "refund_amount": np.where(rng.random(n) < 0.08, rng.uniform(10, 500, n).round(2), 0),
        "payment_failures": rng.integers(0, 4, n),
        "days_overdue": rng.integers(0, 45, n),
        "bill_shock_flag": (monthly_charges > 800).astype(int),
        "paperless_billing": rng.integers(0, 2, n),
        "plan_upgrades": rng.integers(0, 3, n),
        "reactivation_count": rng.integers(0, 2, n),
        "churn": churn,
    })

    # Inject nulls (~4% on selected numerics)
    null_cols = ["data_volume_mb", "call_minutes", "refund_amount", "latency_ms", "csat_score", "signal_strength_dbm"]
    for col in null_cols:
        mask = rng.random(n) < 0.04
        df.loc[mask, col] = np.nan

    # Bad tokens
    bad_idx = rng.choice(n, 40, replace=False)
    df.loc[bad_idx[:15], "monthly_charges"] = "-"
    df.loc[bad_idx[15:25], "payment_method"] = "N/A"
    df.loc[bad_idx[25:30], "contract_type"] = "null"
    df.loc[bad_idx[30:35], "service_status"] = "None"
    df.loc[bad_idx[35:], "plan"] = "  "

    # Duplicates
    dup = df.iloc[:10].copy()
    df = pd.concat([df, dup], ignore_index=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def step_ingest() -> None:
    _step_header(
        1, "Data Ingestion",
        "Upload your telecom dataset (CSV / Excel), download the ready-made sample CSV to test "
        "all checklist items, or click **Use Demo Data** to load a synthetic 2,000-subscriber "
        "dataset directly.",
    )

    with st.expander("📋  Full Field Dictionary — 42 Telco Columns", expanded=False):
        fd = pd.DataFrame(FIELD_DICT, columns=["Column", "Category", "Type", "Description"])
        st.dataframe(fd, use_container_width=True, hide_index=True, height=420)
        cat_summary = (
            fd.groupby("Category")["Column"].count().reset_index(name="Fields")
        )
        fig_cat = px.bar(
            cat_summary, x="Category", y="Fields", text="Fields",
            title="Fields by Category", color="Fields", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("✅  Pipeline Checklist — What each field tests", expanded=False):
        cl = pd.DataFrame(PIPELINE_CHECKLIST, columns=["Step", "Field(s) Used", "What is Tested"])
        st.dataframe(cl, use_container_width=True, hide_index=True)

    col_up, col_demo, col_dl = st.columns([2, 1, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
        )
    with col_demo:
        if st.button("Use Demo Data", type="primary", use_container_width=True):
            df = _demo_data(2000)
            ss.raw_df = df
            ss.dataset_name = "Telco Demo Dataset (2,000 rows × 42 cols)"
            ss.demo = True
            ss.target_col = "churn" if "churn" in df.columns else df.columns[-1]
            ss.cleaned_df = None
            ss.featured_df = None
            ss.model_results = {}
            ss.models = {}
            _ok(f"Demo dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    with col_dl:
        if SAMPLE_CSV_PATH.exists():
            st.download_button(
                "Download Sample CSV",
                data=SAMPLE_CSV_PATH.read_bytes(),
                file_name="telco_sample_data.csv",
                mime="text/csv",
                use_container_width=True,
                help="500-row CSV with all 42 fields — upload it back to test every step",
            )
        else:
            st.caption("Sample CSV not found")

    if uploaded is not None:
        try:
            name = uploaded.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            ss.raw_df = df
            ss.dataset_name = uploaded.name
            ss.demo = False
            if "churn" in df.columns:
                ss.target_col = "churn"
            ss.cleaned_df = None
            ss.featured_df = None
            ss.model_results = {}
            ss.models = {}
            _ok(f"Loaded **{uploaded.name}**: {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    if ss.raw_df is None:
        st.info("Load data above to continue.")
        return

    df = ss.raw_df
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True, height=320)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Total Records", f"{df.shape[0]:,}")
    with c2:
        _kpi("Columns", df.shape[1], "green")
    with c3:
        _kpi("Missing Cells", f"{int(df.isnull().sum().sum()):,}", "orange")
    with c4:
        _kpi("Duplicates", f"{int(df.duplicated().sum()):,}", "purple")

    st.subheader("Column Inventory")
    inv = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null %": (df.isnull().mean() * 100).round(1).values,
        "Unique": df.nunique().values,
        "Sample": [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "" for c in df.columns],
    })
    st.dataframe(inv, use_container_width=True, hide_index=True)

    st.subheader("Target Column (for Churn Prediction)")
    tgt_options = list(df.columns)
    default_idx = tgt_options.index(ss.target_col) if ss.target_col in tgt_options else 0
    ss.target_col = st.selectbox(
        "Select the churn / target column (must be 0/1 or binary):",
        tgt_options,
        index=default_idx,
    )

    target_series = df[ss.target_col].dropna()
    if target_series.dtype == object:
        unique_vals = sorted(target_series.unique())
        _warn(
            f"Target `{ss.target_col}` is text. Values: {unique_vals[:5]}. "
            "It will be label-encoded in the Feature Engineering step."
        )
    elif target_series.nunique() > 2:
        _warn(
            f"Target `{ss.target_col}` has {target_series.nunique()} distinct values "
            f"({sorted(target_series.unique())[:8]}…). "
            "Churn prediction requires a **binary** column (0/1)."
        )

    if st.button("Confirm Ingestion & Proceed →", type="primary"):
        _complete_step(1)
        st.rerun()





# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def step_eda() -> None:
    _step_header(
        2, "Exploratory Data Analysis",
        "Understand distributions, correlations, missing values, and target balance before cleaning.",
    )
    if ss.raw_df is None:
        st.warning("Complete Step 1 first.")
        return

    df = ss.raw_df.copy()
    tgt = ss.target_col
    ss.eda_insights = []

    st.subheader("Descriptive Statistics")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.dataframe(df[num_cols].describe().T.round(3), use_container_width=True)
    else:
        st.caption("No numeric columns detected.")

    st.subheader(f"Target Distribution — {tgt}")
    if tgt in df.columns:
        vc = df[tgt].value_counts(dropna=False).reset_index()
        vc.columns = [tgt, "count"]
        vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(1).astype(str) + "%"
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig1 = px.pie(vc, names=tgt, values="count", title="Class Proportions")
            st.plotly_chart(fig1, use_container_width=True)
        with col_bar:
            bar_color = tgt if vc.shape[0] < 10 else None
            fig2 = px.bar(vc, x=tgt, y="count", text="pct", color=bar_color, title="Class Counts")
            st.plotly_chart(fig2, use_container_width=True)
        churn_rate = df[tgt].astype(float).mean() if pd.to_numeric(df[tgt], errors="coerce").notna().any() else None
        if churn_rate is not None:
            _insight(f"Overall churn / positive-class rate: **{churn_rate:.1%}**")
            ss.eda_insights.append(f"Churn rate: {churn_rate:.1%}")
    else:
        st.error(f"Target column `{tgt}` not found.")

    st.subheader("Correlation Heatmap")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig3 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Matrix",
        )
        st.plotly_chart(fig3, use_container_width=True)
        if tgt in num_cols:
            top_corr = corr[tgt].drop(tgt, errors="ignore").abs().sort_values(ascending=False).head(5)
            if len(top_corr):
                pairs = ", ".join(f"`{k}` ({v:.2f})" for k, v in top_corr.items())
                _insight(f"Top correlations with `{tgt}`: {pairs}")
    else:
        st.caption("Need at least 2 numeric columns for correlation.")

    st.subheader("Missing Values")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        miss_df = miss.reset_index()
        miss_df.columns = ["Column", "Missing"]
        miss_df["Pct"] = (miss_df["Missing"] / len(df) * 100).round(1)
        fig4 = px.bar(miss_df, x="Column", y="Missing", color="Pct",
                      color_continuous_scale="Oranges", title="Missing Value Counts")
        st.plotly_chart(fig4, use_container_width=True)
        _warn(f"{len(miss)} columns contain null values — address in Step 3.")
        ss.eda_insights.append(f"{len(miss)} columns with missing values")
    else:
        _ok("No missing values detected in raw data.")

    st.subheader("Numeric Distributions")
    if num_cols:
        dist_col = st.selectbox("Numeric column for distribution:", num_cols, key="eda_dist_col")
        if dist_col:
            fig_dist = px.histogram(
                df, x=dist_col, color=tgt if tgt in df.columns else None,
                nbins=40, marginal="box", title=f"Distribution — {dist_col}",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            skew = df[dist_col].skew()
            _insight(f"`{dist_col}` skewness: **{skew:.2f}**" + (" (right-skewed)" if skew > 1 else ""))

    st.subheader("Target Correlation Ranking")
    if tgt in num_cols:
        tgt_corr = (
            df[num_cols].corr()[tgt].drop(tgt, errors="ignore").abs().sort_values(ascending=False)
        )
        corr_df = tgt_corr.head(12).reset_index()
        corr_df.columns = ["Feature", "|Correlation|"]
        fig_tc = px.bar(corr_df, x="|Correlation|", y="Feature", orientation="h",
                        title=f"Top Features Correlated with {tgt}")
        st.plotly_chart(fig_tc, use_container_width=True)

    st.subheader("Categorical Breakdown")
    cat_cols = [c for c in df.select_dtypes("object").columns if c != tgt][:4]
    if cat_cols and tgt in df.columns:
        pick = st.selectbox("Categorical column to explore:", cat_cols)
        if pick:
            ct = df.groupby(pick, dropna=False)[tgt].apply(
                lambda s: pd.to_numeric(s, errors="coerce").mean()
            ).reset_index()
            ct.columns = [pick, "churn_rate"]
            fig5 = px.bar(ct.sort_values("churn_rate", ascending=False),
                          x=pick, y="churn_rate", title=f"Churn Rate by {pick}")
            st.plotly_chart(fig5, use_container_width=True)

    if st.button("Complete EDA & Proceed →", type="primary"):
        _complete_step(2)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def step_clean() -> None:
    _step_header(
        3, "Data Cleaning",
        "Remove duplicates, normalise bad tokens, impute missing values, and coerce types.",
    )
    if ss.raw_df is None:
        st.warning("Complete Step 1 first.")
        return

    df = ss.raw_df.copy()
    log: list[str] = []

    st.subheader("Cleaning Options")
    c1, c2, c3, c4 = st.columns(4)
    drop_dup = c1.checkbox("Remove duplicate rows", value=True)
    fix_null_tokens = c2.checkbox("Normalise null tokens (N/A, null, None)", value=True)
    strip_blanks = c3.checkbox("Strip blanks → null", value=True)
    fix_types = c4.checkbox("Auto-coerce numeric columns", value=True)
    impute = st.checkbox("Impute missing numeric values (median)", value=True)

    if st.button("Run Cleaning Pipeline", type="primary"):
        # 1. Strip whitespace on object columns
        if strip_blanks:
            for c in df.select_dtypes("object").columns:
                df[c] = df[c].astype(str).str.strip()
                blanks = df[c].isin(["", "nan", "NaN"])
                if blanks.any():
                    df.loc[blanks, c] = np.nan
                    log.append(f"Blank strings in `{c}` → null ({int(blanks.sum())} cells).")

        # 2. Null-token normalisation
        if fix_null_tokens:
            null_tokens = {"N/A", "n/a", "NA", "null", "NULL", "None", "none", "-"}
            for c in df.select_dtypes("object").columns:
                mask = df[c].isin(null_tokens)
                if mask.any():
                    df.loc[mask, c] = np.nan
                    log.append(f"Null tokens in `{c}` normalised ({int(mask.sum())} cells).")

        # 3. Duplicates
        if drop_dup:
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            removed = before - len(df)
            if removed:
                log.append(f"Removed {removed} duplicate rows.")

        # 4. Impute numerics (first pass)
        num_cols = df.select_dtypes(include="number").columns
        if impute and len(num_cols):
            imputer = SimpleImputer(strategy="median")
            df[num_cols] = imputer.fit_transform(df[num_cols])
            log.append(f"Median-imputed {len(num_cols)} numeric columns.")

        # 5. Coerce bad numeric tokens (e.g. monthly_charges '-')
        if fix_types:
            for c in df.select_dtypes("object").columns:
                if c == ss.target_col:
                    continue
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notnull().mean() > 0.9:
                    df[c] = converted
                    log.append(f"Auto-coerced `{c}` to numeric.")

        # 6. Re-impute numerics after coercion
        num_cols = df.select_dtypes(include="number").columns
        if impute and len(num_cols) and df[num_cols].isnull().any().any():
            imputer = SimpleImputer(strategy="median")
            df[num_cols] = imputer.fit_transform(df[num_cols])
            log.append("Re-imputed numeric columns after type coercion.")

        ss.cleaned_df = df
        ss.clean_log = log
        _ok(f"Cleaning complete — {len(df):,} rows remain.")

    if ss.cleaned_df is not None:
        st.subheader("Cleaning Log")
        for entry in ss.clean_log:
            _ok(entry)

        st.subheader("Before vs After")
        raw = ss.raw_df
        clean = ss.cleaned_df
        compare = pd.DataFrame({
            "Metric": ["Rows", "Columns", "Missing Cells", "Duplicates"],
            "Before": [
                raw.shape[0], raw.shape[1],
                int(raw.isnull().sum().sum()), int(raw.duplicated().sum()),
            ],
            "After": [
                clean.shape[0], clean.shape[1],
                int(clean.isnull().sum().sum()), int(clean.duplicated().sum()),
            ],
        })
        compare["Delta"] = compare["After"] - compare["Before"]
        st.dataframe(compare, use_container_width=True, hide_index=True)

        st.subheader("Cleaned Data Preview")
        st.dataframe(ss.cleaned_df.head(15), use_container_width=True)
        if st.button("Confirm Cleaning & Proceed →", type="primary"):
            _complete_step(3)
            st.rerun()
    else:
        st.info("Click **Run Cleaning Pipeline** to process the data.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def step_features() -> None:
    _step_header(
        4, "Feature Engineering",
        "Derived features, encoding, and scaling are applied automatically. "
        "Select which columns to include as model features.",
    )
    src = ss.cleaned_df if ss.cleaned_df is not None else ss.raw_df
    if src is None:
        st.warning("Complete Steps 1–3 first.")
        return

    df = src.copy()
    tgt = ss.target_col
    derived: list[str] = []

    st.subheader("Derived KPI Features")

    if "monthly_charges" in df.columns and "data_volume_mb" in df.columns:
        df["charge_per_mb"] = (df["monthly_charges"] / df["data_volume_mb"].replace(0, np.nan)).round(4)
        derived.append("charge_per_mb = monthly_charges / data_volume_mb")

    if "complaint_count" in df.columns and "tenure_months" in df.columns:
        df["complaint_rate"] = (
            df["complaint_count"] / df["tenure_months"].replace(0, np.nan)
        ).round(4)
        derived.append("complaint_rate = complaint_count / tenure_months")

    eng_cols = [c for c in ["call_minutes", "sms_count", "data_volume_mb"] if c in df.columns]
    if eng_cols:
        for c in eng_cols:
            std = df[c].std()
            mean = df[c].mean()
            df[f"{c}_z"] = (df[c] - mean) / (std if std > 0 else 1)
        df["engagement_score"] = df[[f"{c}_z" for c in eng_cols]].mean(axis=1).round(4)
        df = df.drop(columns=[f"{c}_z" for c in eng_cols])
        derived.append("engagement_score = mean z-score of call/SMS/data usage")

    if "tenure_months" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"], bins=[0, 6, 12, 24, 48, 999],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48m+"],
        ).astype(str)
        derived.append("tenure_bucket = binned tenure_months")

    if all(c in df.columns for c in ["created_date", "closed_date"]):
        created = pd.to_datetime(df["created_date"], errors="coerce")
        closed = pd.to_datetime(df["closed_date"], errors="coerce")
        df["resolution_hours"] = ((closed - created).dt.total_seconds() / 3600).round(2)
        if "sla_hours" in df.columns:
            df["sla_status"] = np.where(
                df["resolution_hours"] <= df["sla_hours"], "Within SLA", "Breached SLA"
            )
            derived.append("sla_status + resolution_hours from ticket timestamps")

    for d in derived:
        _ok(d)

    st.subheader("Categorical Encoding")
    cat_cols = df.select_dtypes("object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in _SKIP_ENCODE_COLS]

    if cat_cols:
        for c in cat_cols:
            if c == tgt:
                continue
            n_unique = df[c].nunique()
            if n_unique > _MAX_OHE_CARDINALITY:
                _warn(
                    f"Skipped `{c}` — {n_unique} unique values "
                    f"(max {_MAX_OHE_CARDINALITY} for one-hot encoding)."
                )
                continue
            if n_unique <= 2:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                _ok(f"Label-encoded `{c}` ({n_unique} classes)")
            else:
                dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
                df = pd.concat([df.drop(columns=[c]), dummies], axis=1)
                _ok(f"One-hot-encoded `{c}` → {len(dummies.columns)} dummy columns")

    for skip in _SKIP_ENCODE_COLS:
        if skip in df.columns and skip != tgt:
            _warn(f"Excluded `{skip}` from encoding (identifier / date field).")

    st.subheader("Select Model Features")
    exclude = {"subscriber_id", "msisdn", tgt} | _SKIP_ENCODE_COLS
    candidates = [c for c in df.columns if c not in exclude and c != tgt]
    default_sel = [c for c in candidates if c in ss.feature_cols] or candidates[:20]
    selected = st.multiselect("Features to include:", candidates, default=default_sel)

    st.subheader("Feature Scaling")
    scale_method = st.selectbox("Scaling method:", ["StandardScaler", "None"])

    if st.button("Apply Features & Split Data", type="primary"):
        target = _prepare_binary_target(df[tgt].copy(), tgt)
        if target is None:
            return

        valid = target.notna()
        feat_df = df.loc[valid, selected].copy()
        target = target.loc[valid]

        X_train, X_test, y_train, y_test = _safe_train_test_split(feat_df, target)

        if scale_method == "StandardScaler":
            scaler = StandardScaler()
            cols = X_train.columns.tolist()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=cols, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=cols, index=X_test.index)
            _ok("Applied StandardScaler (fit on train, transform test).")

        if X_train.isnull().any().any() or X_test.isnull().any().any():
            imputer = SimpleImputer(strategy="median")
            cols = X_train.columns.tolist()
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)
            _ok("Imputed remaining NaNs (fit on train only).")

        ss.featured_df = df
        ss.feature_cols = selected
        ss.X_train = X_train
        ss.X_test = X_test
        ss.y_train = y_train
        ss.y_test = y_test
        _ok(
            f"Train: {X_train.shape[0]:,} rows · Test: {X_test.shape[0]:,} rows · "
            f"{len(selected)} features"
        )

    if ss.X_train is not None:
        st.subheader("Train / Test Split Summary")
        split_info = pd.DataFrame({
            "Set": ["Train", "Test"],
            "Rows": [len(ss.X_train), len(ss.X_test)],
            "Features": [ss.X_train.shape[1], ss.X_test.shape[1]],
            "Churn Rate": [
                f"{ss.y_train.mean():.1%}",
                f"{ss.y_test.mean():.1%}",
            ],
        })
        st.dataframe(split_info, use_container_width=True, hide_index=True)

        st.subheader("Feature Matrix Preview (Train)")
        st.dataframe(ss.X_train.head(10), use_container_width=True)

        st.subheader("Feature Statistics (Train)")
        st.dataframe(ss.X_train.describe().T.round(3), use_container_width=True, height=320)

        if st.button("Confirm Features & Proceed →", type="primary"):
            _complete_step(4)
            st.rerun()






# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def step_train() -> None:
    _step_header(5, "Model Training")
    st.markdown(
        "Six models are trained and compared: "
        "**Logistic Regression**, **Random Forest**, **Gradient Boosting**, "
        "**Support Vector Machine**, **K-Nearest Neighbors**, and **AdaBoost**."
    )

    if ss.X_train is None:
        st.warning("Complete Step 4 first.")
        return

    with st.expander("Hyperparameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        lr_c = col1.slider("LR: regularisation C", 0.01, 10.0, 1.0, 0.01)
        rf_trees = col2.slider("RF: n_estimators", 50, 500, 200, 50)
        gb_lr = col3.slider("GB: learning_rate", 0.01, 0.5, 0.1, 0.01)
        gb_trees = col3.slider("GB: n_estimators", 50, 300, 100, 50)
        class_wt = col1.checkbox("Use class_weight='balanced'", value=True)

        col4, col5, col6 = st.columns(3)
        svm_c = col4.slider("SVM: regularisation C", 0.1, 10.0, 1.0, 0.1)
        knn_k = col5.slider("KNN: neighbours (k)", 3, 25, 5, 1)
        ada_trees = col6.slider("AdaBoost: n_estimators", 50, 300, 100, 50)

    st.markdown(
        """
| Model | Type | Strengths |
|-------|------|-----------|
| Logistic Regression | Linear | Fast baseline, interpretable coefficients |
| Random Forest | Ensemble (bagging) | Non-linear patterns, feature importance |
| Gradient Boosting | Ensemble (boosting) | Strong tabular performance |
| Support Vector Machine | Kernel classifier | Effective in high-dimensional space |
| K-Nearest Neighbors | Instance-based | Simple, no training phase |
| AdaBoost | Boosting | Combines weak learners adaptively |
"""
    )

    if st.button("Train All Models", type="primary"):
        X_train = ss.X_train.copy()
        X_test = ss.X_test.copy()
        y_train = ss.y_train.copy()
        y_test = ss.y_test.copy()

        if X_train.isnull().any().any() or X_test.isnull().any().any():
            imputer = SimpleImputer(strategy="median")
            cols = X_train.columns.tolist()
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
            X_test = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)
            _warn("NaN values imputed before training (median, fit on train).")

        cw = "balanced" if class_wt else None
        model_defs = {
            "Logistic Regression": LogisticRegression(C=lr_c, max_iter=1000, class_weight=cw),
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_trees, random_state=42, class_weight=cw, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=gb_trees, learning_rate=gb_lr, random_state=42
            ),
            "Support Vector Machine": SVC(C=svm_c, probability=True, class_weight=cw, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=knn_k),
            "AdaBoost": AdaBoostClassifier(n_estimators=ada_trees, random_state=42),
        }

        results = {}
        trained = {}
        with st.spinner("Training models…"):
            for name, model in model_defs.items():
                st.write(f"Training {name}…")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = (
                    model.predict_proba(X_test)[:, 1]
                    if hasattr(model, "predict_proba") else y_pred.astype(float)
                )
                results[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_prob),
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }
                trained[name] = model

        ss.models = trained
        ss.model_results = results

        if "Random Forest" in trained:
            rf = trained["Random Forest"]
            imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            ss.feature_importance = imp

        _ok(f"{len(trained)} models trained successfully!")

    if ss.model_results:
        st.subheader("Training Summary")
        summary = pd.DataFrame({
            name: {k: v for k, v in res.items() if k not in ("y_pred", "y_prob")}
            for name, res in ss.model_results.items()
        }).T.round(4)
        st.dataframe(summary, use_container_width=True)

        best = max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
        _insight(f"Best model by ROC-AUC: **{best}** ({ss.model_results[best]['roc_auc']:.3f})")

        if ss.feature_importance is not None:
            st.subheader("Top Feature Importances (Random Forest)")
            fi = ss.feature_importance.head(15).reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Top 15 Features")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Confirm Training & Proceed →", type="primary"):
            _complete_step(5)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def step_evaluate() -> None:
    _step_header(6, "Model Evaluation", "Compare confusion matrices, ROC curves, and classification reports.")

    if not ss.model_results:
        st.warning("Complete Step 5 first.")
        return

    y_test = ss.y_test
    model_names = list(ss.model_results.keys())
    pick = st.selectbox("Select model for detailed view:", model_names)
    res = ss.model_results[pick]

    st.markdown(f"### {pick} — Performance Card")
    card_cols = st.columns(6)
    card_cols[0].metric("Accuracy", f"{res['accuracy']:.1%}")
    card_cols[1].metric("Precision", f"{res['precision']:.1%}")
    card_cols[2].metric("Recall", f"{res['recall']:.1%}")
    card_cols[3].metric("F1 Score", f"{res['f1']:.1%}")
    card_cols[4].metric("ROC-AUC", f"{res['roc_auc']:.3f}")
    rank = sorted(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"], reverse=True).index(pick) + 1
    card_cols[5].metric("AUC Rank", f"#{rank} / {len(model_names)}")

    st.subheader("Side-by-Side Model Ranking")
    rank_df = pd.DataFrame({
        name: {
            "Accuracy": res["accuracy"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "F1": res["f1"],
            "ROC-AUC": res["roc_auc"],
        }
        for name, res in ss.model_results.items()
    }).T.sort_values("ROC-AUC", ascending=False).round(4)
    rank_df["Rank"] = range(1, len(rank_df) + 1)
    st.dataframe(rank_df, use_container_width=True)

    fig_rank = px.bar(
        rank_df.reset_index().rename(columns={"index": "Model"}),
        x="ROC-AUC", y="Model", orientation="h", color="ROC-AUC",
        color_continuous_scale="Blues", title="Models Ranked by ROC-AUC",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{res['accuracy']:.3f}")
    c2.metric("Precision", f"{res['precision']:.3f}")
    c3.metric("Recall", f"{res['recall']:.3f}")
    c4.metric("F1", f"{res['f1']:.3f}")
    c5.metric("ROC-AUC", f"{res['roc_auc']:.3f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, res["y_pred"])
        fig_cm = px.imshow(
            cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
            color_continuous_scale="Blues", title=f"{pick} — Confusion Matrix",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        st.subheader("ROC Curve — All Models")
        fig_roc = go.Figure()
        for name, r in ss.model_results.items():
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['roc_auc']:.3f})"
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_test, res["y_pred"], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.subheader("Churn Probability Distribution")
    prob_df = pd.DataFrame({"churn_probability": res["y_prob"]})
    fig_hist = px.histogram(prob_df, x="churn_probability", nbins=30, title="P(churn) on Test Set")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Model Comparison Heatmap")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    hm = pd.DataFrame({
        name: [res[m] for m in metrics]
        for name, res in ss.model_results.items()
    }, index=metrics)
    fig_hm = px.imshow(
        hm, text_auto=".3f", aspect="auto",
        color_continuous_scale="Viridis", title="Metrics Heatmap — All Models",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    if st.button("Complete Evaluation & Proceed →", type="primary"):
        _complete_step(6)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BIAS DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def step_bias() -> None:
    _step_header(
        7, "Bias Detection",
        "**Learning curves** show training score vs. cross-validation score as the training size grows.  \n"
        "- **Overfit**: high train score, low val score (gap stays large)  \n"
        "- **Underfit**: both scores are low and close together  \n"
        "- **Good fit**: both scores converge at a high value",
    )

    if not ss.models or ss.X_train is None:
        st.warning("Complete Step 5 first.")
        return

    model_names = list(ss.models.keys())
    pick = st.selectbox("Model for learning curve:", model_names)
    model = ss.models[pick]

    if st.button("Compute Learning Curve", type="primary"):
        with st.spinner("Computing learning curve (may take a moment)…"):
            train_sizes, train_scores, val_scores = learning_curve(
                model, ss.X_train, ss.y_train, cv=3,
                train_sizes=np.linspace(0.2, 1.0, 5), scoring="roc_auc", n_jobs=-1,
            )
        train_mean = train_scores.mean(axis=1)
        val_mean = val_scores.mean(axis=1)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Train AUC"))
        fig.add_trace(go.Scatter(x=train_sizes, y=val_mean, mode="lines+markers", name="CV AUC"))
        fig.update_layout(title=f"Learning Curve — {pick}", xaxis_title="Training Size", yaxis_title="ROC-AUC")
        st.plotly_chart(fig, use_container_width=True)

        gap = train_mean[-1] - val_mean[-1]
        if gap > 0.08:
            diagnosis = "Overfitting"
        elif val_mean[-1] < 0.65:
            diagnosis = "Underfitting"
        else:
            diagnosis = "Good Fit"
        _insight(f"Diagnosis for **{pick}**: **{diagnosis}** (train–CV gap = {gap:.3f})")

    st.subheader("Train vs Test AUC Gap")
    rows = []
    for name, model in ss.models.items():
        if not hasattr(model, "predict_proba"):
            continue
        train_auc = roc_auc_score(ss.y_train, model.predict_proba(ss.X_train)[:, 1])
        test_auc = ss.model_results[name]["roc_auc"]
        gap = train_auc - test_auc
        if gap > 0.08:
            fit = "Overfitting"
        elif test_auc < 0.65:
            fit = "Underfitting"
        else:
            fit = "Good Fit"
        rows.append({"Model": name, "Train AUC": train_auc, "Test AUC": test_auc, "Gap": gap, "Fit": fit})

    if rows:
        gap_df = pd.DataFrame(rows).round(4)
        st.dataframe(gap_df, use_container_width=True, hide_index=True)
        fig_gap = px.bar(gap_df, x="Model", y="Gap", color="Fit", title="Train − Test AUC Gap")
        st.plotly_chart(fig_gap, use_container_width=True)

    if st.button("Complete Bias Check & Proceed →", type="primary"):
        _complete_step(7)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_CAMPAIGNS = {
    "High-Risk / Service Issue": {
        "product": "Priority Support Plan",
        "campaign": "Service Recovery — Personal apology + 3-month fee waiver",
        "urgency": "CRITICAL",
        "color": "reco-red",
    },
    "High-Risk / Low Usage": {
        "product": "Starter Data Bundle 2 GB @ R29",
        "campaign": "Win-Back — Free 1GB data for 2 months",
        "urgency": "HIGH",
        "color": "reco-amber",
    },
    "High-Risk / High Value": {
        "product": "Loyalty Rewards Programme",
        "campaign": "Retention — 20% discount + dedicated account manager",
        "urgency": "HIGH",
        "color": "reco-amber",
    },
    "At-Risk": {
        "product": "Flexi Bundle (voice + data + SMS)",
        "campaign": "Proactive Engagement — Usage milestone reward",
        "urgency": "MEDIUM",
        "color": "reco-amber",
    },
    "New Subscriber": {
        "product": "Welcome Pack + 30-day trial upgrade",
        "campaign": "Onboarding Journey — Guided tutorials + first-month bonus",
        "urgency": "MEDIUM",
        "color": "reco-blue",
    },
    "Loyal / High Value": {
        "product": "5G Early Access / Premium Plan Upgrade",
        "campaign": "Upsell — Early 5G access + referral bonus",
        "urgency": "LOW",
        "color": "reco-green",
    },
    "Stable / Standard": {
        "product": "Data Add-On or SMS Bundle",
        "campaign": "Cross-Sell — Personalised data top-up offers",
        "urgency": "LOW",
        "color": "reco-green",
    },
}


def step_recommendations() -> None:
    _step_header(
        8, "Recommendations",
        "Score every subscriber for churn probability and assign one of seven actionable segments.",
    )

    if not ss.model_results or ss.X_test is None:
        st.warning("Complete Steps 5–6 first.")
        return

    best = max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
    model = ss.models[best]

    def _segment(row) -> str:
        p = row["churn_probability"]
        complaints = row.get("complaint_count", 0) or 0
        data_mb = row.get("data_volume_mb", 0) or 0
        charges = row.get("monthly_charges", 0) or 0
        tenure = row.get("tenure_months", 12) or 12

        if p >= 0.65 and complaints >= 3:
            return "High-Risk / Service Issue"
        if p >= 0.65 and data_mb < 200:
            return "High-Risk / Low Usage"
        if p >= 0.65 and charges > 600:
            return "High-Risk / High Value"
        if p >= 0.4:
            return "At-Risk"
        if tenure < 6:
            return "New Subscriber"
        if data_mb > 3000 or charges > 500:
            return "Loyal / High Value"
        return "Stable / Standard"

    if st.button("Generate Recommendations", type="primary"):
        probs = model.predict_proba(ss.X_test)[:, 1]
        reco = ss.X_test.copy()
        reco["churn_probability"] = probs.round(4)

        src = ss.cleaned_df if ss.cleaned_df is not None else ss.raw_df
        meta_cols = [c for c in [
            "complaint_count", "data_volume_mb", "monthly_charges", "tenure_months",
        ] if src is not None and c in src.columns]
        for c in meta_cols:
            if c in reco.columns:
                continue
            if c in ss.X_test.columns:
                reco[c] = ss.X_test[c].values
            elif src is not None:
                aligned = src.reindex(reco.index)
                reco[c] = aligned[c].values

        reco["segment"] = reco.apply(_segment, axis=1)
        for field in ("product", "campaign", "urgency", "color"):
            reco[field] = reco["segment"].map(lambda s, f=field: SEGMENT_CAMPAIGNS[s][f])

        ss.recommendations_df = reco
        high_risk = int((reco["churn_probability"] >= 0.5).sum())
        _ok(f"Scored {len(reco):,} subscribers — {high_risk:,} high-risk (P ≥ 50%).")

    if ss.recommendations_df is not None:
        reco = ss.recommendations_df
        st.subheader("Customer Segmentation")
        seg_counts = reco["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(seg_counts, names="Segment", values="Count", title="Segment Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            seg_avg = reco.groupby("segment")["churn_probability"].mean().reset_index()
            seg_avg.columns = ["segment", "Avg P(churn)"]
            fig_bar = px.bar(
                seg_avg.sort_values("Avg P(churn)", ascending=True),
                x="Avg P(churn)", y="segment", orientation="h",
                color="Avg P(churn)", color_continuous_scale="Reds",
                title="Average Churn Probability by Segment",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Scored Subscribers Preview")
        preview_cols = [
            c for c in [
                "churn_probability", "segment", "complaint_count",
                "data_volume_mb", "monthly_charges", "tenure_months",
                "product", "campaign", "urgency",
            ] if c in reco.columns
        ]
        st.dataframe(
            reco[preview_cols].sort_values("churn_probability", ascending=False).head(25),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Campaign Recommendations by Segment")
        for seg in seg_counts["Segment"]:
            sub = reco[reco["segment"] == seg]
            camp = SEGMENT_CAMPAIGNS[seg]
            st.markdown(
                f'<div class="reco-card {camp["color"]}"><strong>{seg}</strong> &nbsp;·&nbsp; '
                f'{len(sub):,} subscribers  <br/>📦 <em>Product:</em> {camp["product"]}'
                f'  <br/>📢 <em>Campaign:</em> {camp["campaign"]}'
                f'  <br/>🔴 <em>Urgency:</em> {camp["urgency"]}</div>',
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download Recommendation CSV",
            data=reco.to_csv(index=False).encode(),
            file_name="churn_recommendations.csv",
            mime="text/csv",
        )

        if st.button("Proceed to Data Story →", type="primary"):
            _complete_step(8)
            st.rerun()






# ─────────────────────────────────────────────────────────────────────────────
# POWERPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ppt_bullet_slide(prs, title: str, bullets: list[str]) -> None:
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    tf = slide.placeholders[1].text_frame
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(18)


def _ppt_table_slide(prs, title: str, headers: list[str], rows: list[list]) -> None:
    from pptx.util import Inches, Pt

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    nrows = len(rows) + 1
    ncols = len(headers)
    left, top, width, height = Inches(0.5), Inches(1.5), Inches(12.3), Inches(0.4 * nrows)
    table = slide.shapes.add_table(nrows, ncols, left, top, width, height).table
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = str(h)
        cell.text_frame.paragraphs[0].font.bold = True
        cell.text_frame.paragraphs[0].font.size = Pt(12)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            table.cell(i, j).text = str(val)


def _generate_pptx() -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    df = ss.raw_df
    clean_df = ss.cleaned_df
    tgt = ss.target_col
    best = (
        max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
        if ss.model_results else None
    )

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Telecom Subscriber Analytics"
    sub = slide.placeholders[1].text_frame
    sub.text = (
        f"Churn Prediction — Executive Data Story\n"
        f"{ss.dataset_name}  ·  {datetime.now().strftime('%B %Y')}"
    )

    churn_rate = None
    if df is not None and tgt in df.columns:
        churn_rate = pd.to_numeric(df[tgt], errors="coerce").mean()

    bullets = [
        "End-to-end analytics pipeline covering 9 stages:",
        f"Dataset: {ss.dataset_name}",
    ]
    if df is not None:
        bullets.append(f"{df.shape[0]:,} records, {df.shape[1]} features")
    if churn_rate is not None:
        bullets.append(f"Overall churn rate: {churn_rate:.1%}")
    if clean_df is not None:
        bullets.append(f"Clean records after preprocessing: {clean_df.shape[0]:,}")
    if ss.feature_cols:
        bullets.append(f"Model features engineered: {len(ss.feature_cols)}")
    if best:
        bullets.append(
            f"Best model: {best} (ROC-AUC = {ss.model_results[best]['roc_auc']:.3f})"
        )
    if ss.recommendations_df is not None:
        hi = int((ss.recommendations_df["churn_probability"] >= 0.5).sum())
        bullets.append(f"High-risk subscribers (P ≥ 50%): {hi:,}")
    _ppt_bullet_slide(prs, "Executive Summary", bullets)

    if df is not None:
        _ppt_bullet_slide(prs, "Data Quality & Preparation", [
            f"Raw records: {df.shape[0]:,}",
            f"Null rate: {df.isnull().mean().mean():.1%}",
            f"Duplicate rows: {int(df.duplicated().sum()):,}",
            f"Target column: {tgt}",
        ])

    if churn_rate is not None:
        churned = int(pd.to_numeric(df[tgt], errors="coerce").sum())
        retained = int(len(df) - churned)
        status = (
            "CRITICAL — above 25% threshold" if churn_rate > 0.25
            else "ELEVATED — proactive retention recommended" if churn_rate > 0.15
            else "Within acceptable range"
        )
        _ppt_bullet_slide(prs, "Churn Profile", [
            f"Churn rate: {churn_rate:.1%}",
            f"Churned subscribers: {churned:,}",
            f"Retained subscribers: {retained:,}",
            f"Status: {status}",
        ])

    if ss.model_results:
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Fit"]
        rows = []
        for name, res in ss.model_results.items():
            train_auc = None
            if name in ss.models and hasattr(ss.models[name], "predict_proba"):
                train_auc = roc_auc_score(ss.y_train, ss.models[name].predict_proba(ss.X_train)[:, 1])
            gap = (train_auc - res["roc_auc"]) if train_auc else 0
            fit = "Overfitting" if gap > 0.08 else ("Underfitting" if res["roc_auc"] < 0.65 else "Good Fit")
            rows.append([
                name,
                f"{res['accuracy']:.3f}",
                f"{res['precision']:.3f}",
                f"{res['recall']:.3f}",
                f"{res['f1']:.3f}",
                f"{res['roc_auc']:.3f}",
                fit,
            ])
        _ppt_table_slide(prs, "Model Benchmark — Six Classifiers", headers, rows)

    if best:
        res = ss.model_results[best]
        _ppt_bullet_slide(prs, "Recommended Model", [
            f"Recommended model: {best}",
            f"Test ROC-AUC: {res['roc_auc']:.3f}",
            f"Accuracy: {res['accuracy']:.3f}  |  F1: {res['f1']:.3f}",
            f"Precision: {res['precision']:.3f}  |  Recall: {res['recall']:.3f}",
            "Deploy this model for CRM churn scoring and campaign prioritisation.",
        ])

    if ss.feature_importance is not None:
        top = ss.feature_importance.head(8)
        bullets = [f"{feat}: {imp:.4f}" for feat, imp in top.items()]
        _ppt_bullet_slide(prs, "Top Churn Drivers", bullets)

    if ss.recommendations_df is not None:
        seg = ss.recommendations_df.groupby("segment").agg(
            count=("churn_probability", "count"),
            mean=("churn_probability", "mean"),
        ).sort_values("mean", ascending=False)
        bullets = [
            f"{idx}: {int(row['count']):,} subs, avg P(churn) = {row['mean']:.2f}"
            for idx, row in seg.iterrows()
        ]
        _ppt_bullet_slide(prs, "Campaign Recommendations", bullets)

    _ppt_bullet_slide(prs, "Recommended Action Plan", [
        "Immediate (0–7 days): Contact high-risk / service-issue subscribers; escalate complaints.",
        "Short-term (1–4 weeks): Launch win-back data bundle campaign; enable CRM churn alerts.",
        "Medium-term (1–3 months): Cross-sell flexi bundles; automate onboarding journeys.",
        "Long-term (3–12 months): Premium 5G upgrades for loyal segments; real-time ML scoring.",
    ])

    if ss.model_results:
        _ppt_table_slide(
            prs,
            "Quick Model Comparison",
            ["Model", "ROC-AUC", "F1", "Accuracy"],
            [
                [
                    name,
                    f"{res['roc_auc']:.3f}",
                    f"{res['f1']:.3f}",
                    f"{res['accuracy']:.3f}",
                ]
                for name, res in sorted(
                    ss.model_results.items(),
                    key=lambda x: x[1]["roc_auc"],
                    reverse=True,
                )
            ],
        )

    if ss.recommendations_df is not None:
        urgent = ss.recommendations_df[
            ss.recommendations_df["urgency"].isin(["CRITICAL", "HIGH"])
        ]
        _ppt_bullet_slide(prs, "Priority Outreach Queue", [
            f"Critical + high urgency subscribers: {len(urgent):,}",
            f"Average P(churn) in priority queue: {urgent['churn_probability'].mean():.2f}",
            "Focus: High-Risk / Service Issue and High-Risk / High Value segments first.",
            "CRM action: assign dedicated retention agent within 48 hours.",
        ])

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Thank You"
    slide.placeholders[1].text_frame.text = (
        "Questions & Next Steps\n\n"
        "Integrate model scores into CRM  ·  Monitor monthly  ·  Retrain quarterly"
    )

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _generate_technical_pptx() -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt

    df = ss.raw_df
    clean_df = ss.cleaned_df
    tgt = ss.target_col
    best = (
        max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
        if ss.model_results else None
    )

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = "Technical Deep Dive"
    sub = slide.placeholders[1].text_frame
    sub.text = (
        f"Telco Churn ML Pipeline — Technical Documentation\n"
        f"{ss.dataset_name}  ·  {datetime.now().strftime('%B %Y')}"
    )

    _ppt_bullet_slide(prs, "Problem Definition", [
        "Business objective: predict subscriber churn and prioritise retention actions.",
        "ML task: binary classification (churn = 1, retained = 0).",
        "Constraints: interpretable features, CRM-ready scores, segment-level actions.",
        f"Dataset: {ss.dataset_name} ({df.shape[0]:,} rows)" if df is not None else "Dataset: N/A",
    ])

    _ppt_bullet_slide(prs, "Pipeline Architecture (9 Steps)", [
        "Step 1 — Data Ingestion: CSV upload / demo dataset (42 telco fields).",
        "Step 2 — EDA: distributions, correlations, class balance, missing-value audit.",
        "Step 3 — Cleaning: dedup, null imputation, type coercion, token normalisation.",
        "Step 4 — Feature Engineering: derived KPIs, encoding, train/test split, scaling.",
        "Step 5 — Training: six classifiers with hyperparameter tuning.",
        "Step 6 — Evaluation: confusion matrix, ROC-AUC, precision/recall/F1.",
        "Step 7 — Bias Detection: learning curves, train vs test AUC gap.",
        "Step 8 — Recommendations: probability scoring + rule-based segmentation.",
        "Step 9 — Data Story: executive + technical exports.",
    ])

    if df is not None:
        _ppt_bullet_slide(prs, "Data Schema", [
            f"Columns: {df.shape[1]} (identity, subscription, usage, network, CX, billing, loyalty)",
            f"Target: {tgt}",
            "Identity keys excluded from modelling: subscriber_id, msisdn, id_number, account_number.",
            "Date fields excluded from one-hot encoding: activation_date, created_date, closed_date.",
            f"High-cardinality skip threshold: {_MAX_OHE_CARDINALITY} unique values.",
        ])

    if ss.clean_log:
        _ppt_bullet_slide(prs, "Cleaning Transformations", ss.clean_log[:12])

    _ppt_bullet_slide(prs, "Feature Engineering Details", [
        "Derived KPI — charge_per_mb = monthly_charges / data_volume_mb",
        "Derived KPI — complaint_rate = complaint_count / tenure_months",
        "Derived KPI — engagement_score = mean z-score(call_minutes, sms_count, data_volume_mb)",
        "Derived KPI — tenure_bucket = pd.cut(tenure_months, [0,6,12,24,48,999])",
        "Categorical encoding: label-encode (≤2 classes) or one-hot (>2, ≤30 cardinality)",
        "Scaling: StandardScaler (optional) fit on train, applied to test.",
        "Target validation: binary 0/1 coercion with stratified split when possible.",
    ])

    algo_notes = [
        ("Logistic Regression", "Linear baseline; fast, interpretable coefficients."),
        ("Random Forest", "Bagged trees; handles non-linearity; feature importance."),
        ("Gradient Boosting", "Sequential boosted trees; strong tabular performance."),
        ("Support Vector Machine", "Kernel margin classifier; probability via Platt scaling."),
        ("K-Nearest Neighbors", "Instance-based; local pattern matching."),
        ("AdaBoost", "Adaptive boosting of weak learners."),
    ]
    _ppt_table_slide(
        prs, "Model Zoo — Six Classifiers",
        ["Model", "Algorithm Notes"],
        [[n, d] for n, d in algo_notes],
    )

    if ss.model_results:
        headers = ["Model", "Train AUC", "Test AUC", "Gap", "Diagnosis"]
        rows = []
        for name, res in ss.model_results.items():
            train_auc = ""
            gap = ""
            diag = ""
            if name in ss.models and hasattr(ss.models[name], "predict_proba"):
                ta = roc_auc_score(ss.y_train, ss.models[name].predict_proba(ss.X_train)[:, 1])
                train_auc = f"{ta:.3f}"
                gap = f"{ta - res['roc_auc']:.3f}"
                diag = "Overfitting" if ta - res["roc_auc"] > 0.08 else (
                    "Underfitting" if res["roc_auc"] < 0.65 else "Good Fit"
                )
            rows.append([name, train_auc, f"{res['roc_auc']:.3f}", gap, diag])
        _ppt_table_slide(prs, "Train vs Test AUC Benchmark", headers, rows)

    _ppt_bullet_slide(prs, "Metrics Glossary", [
        "Accuracy — overall correct predictions.",
        "Precision — of predicted churners, how many actually churned.",
        "Recall — of actual churners, how many were caught.",
        "F1 — harmonic mean of precision and recall.",
        "ROC-AUC — rank quality of probability scores across thresholds.",
    ])

    _ppt_bullet_slide(prs, "Bias & Learning Curves", [
        "Learning curves plot train vs CV score across training sizes.",
        "Large train–CV gap → overfitting; both low → underfitting.",
        "Monitor monthly PSI on features, ROC-AUC drift, prediction volume by segment.",
        "A/B test retention campaigns by segment before full rollout.",
    ])

    if ss.feature_importance is not None:
        headers = ["Feature", "Importance"]
        rows = [[f, f"{v:.4f}"] for f, v in ss.feature_importance.head(12).items()]
        _ppt_table_slide(prs, "Feature Importance (Random Forest — Gini Decrease)", headers, rows)

    _ppt_bullet_slide(prs, "Scoring & Segmentation Rules", [
        "Rule engine maps probability + usage KPIs to 7 segments:",
        "  P ≥ 0.65 & complaints ≥ 3  →  High-Risk / Service Issue",
        "  P ≥ 0.65 & low data usage  →  High-Risk / Low Usage",
        "  P ≥ 0.65 & charges > R600  →  High-Risk / High Value",
        "  P ≥ 0.40                   →  At-Risk",
        "  tenure < 6 months          →  New Subscriber",
        "  high usage / charges       →  Loyal / High Value  |  else Stable / Standard",
    ])

    if best:
        _ppt_bullet_slide(prs, "Model Selection Rationale", [
            f"Selected model: {best}",
            f"Test ROC-AUC: {ss.model_results[best]['roc_auc']:.3f}",
            "Criteria: highest test ROC-AUC among six candidates.",
            "Production: batch score daily; expose probability + segment to CRM API.",
        ])

    _ppt_bullet_slide(prs, "Deployment Considerations", [
        "Retrain quarterly or when AUC drops > 5 points.",
        "Store model artefact + scaler + feature list versioned in MLflow.",
        "Log predictions with SHAP/feature contributions for explainability.",
        "Alert when segment distribution shifts beyond historical bounds.",
    ])

    if ss.eda_insights:
        _ppt_bullet_slide(prs, "EDA Key Findings", ss.eda_insights[:8])

    if ss.recommendations_df is not None:
        seg_detail = ss.recommendations_df.groupby("segment").agg(
            count=("churn_probability", "count"),
            avg_prob=("churn_probability", "mean"),
            max_prob=("churn_probability", "max"),
        ).reset_index()
        headers = ["Segment", "Count", "Avg P(churn)", "Max P(churn)"]
        rows = [
            [r["segment"], int(r["count"]), f"{r['avg_prob']:.3f}", f"{r['max_prob']:.3f}"]
            for _, r in seg_detail.iterrows()
        ]
        _ppt_table_slide(prs, "Segment Detail Table", headers, rows)

    _ppt_bullet_slide(prs, "Reproducibility & Governance", [
        "Random seed: 42 for split and tree-based models.",
        "Train/test split: 80/20 with stratification when class counts allow.",
        "Excluded identifiers: subscriber_id, msisdn, id_number, account_number.",
        "Excluded dates from encoding: activation_date, created_date, closed_date.",
        "Pipeline version exported as JSON alongside PowerPoint decks.",
    ])

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()






# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — DATA STORY
# ─────────────────────────────────────────────────────────────────────────────
def step_story() -> None:
    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#1565c0,#0d47a1);
    color:#fff;border-radius:16px;padding:28px 32px;margin-bottom:24px;">
  <h2 style="margin:0 0 6px 0;">📡 Telecom Subscriber Analytics</h2>
  <p style="margin:0;opacity:.85;font-size:1rem;">
    Executive Data Story — {ss.dataset_name} &nbsp;·&nbsp; {datetime.now().strftime("%B %Y")}
    &nbsp;·&nbsp; Download the PowerPoint deck below for stakeholder presentations
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if ss.raw_df is None:
        st.warning("Complete earlier steps first.")
        return

    df = ss.raw_df
    tgt = ss.target_col
    churn_rate = pd.to_numeric(df[tgt], errors="coerce").mean() if tgt in df.columns else None

    st.subheader("Executive KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        _kpi("Subscribers", f"{df.shape[0]:,}")
    with k2:
        _kpi("Features", len(ss.feature_cols) if ss.feature_cols else df.shape[1], "green")
    with k3:
        _kpi("Churn Rate", f"{churn_rate:.1%}" if churn_rate is not None else "N/A", "orange")
    with k4:
        best = (
            max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
            if ss.model_results else None
        )
        _kpi("Best ROC-AUC", f"{ss.model_results[best]['roc_auc']:.3f}" if best else "N/A", "purple")
    with k5:
        hi = (
            int((ss.recommendations_df["churn_probability"] >= 0.5).sum())
            if ss.recommendations_df is not None else 0
        )
        _kpi("High-Risk Subs", f"{hi:,}", "orange")

    st.subheader("Pipeline Journey")
    journey = []
    step_notes = {
        1: "CSV uploaded or demo data generated",
        2: "EDA insights captured" if ss.eda_insights else "Pending analysis",
        3: f"{len(ss.clean_log)} cleaning steps applied" if ss.clean_log else "Pending cleaning",
        4: f"{len(ss.feature_cols)} features selected" if ss.feature_cols else "Pending engineering",
        5: f"{len(ss.models)} models trained" if ss.models else "Pending training",
        6: "Evaluation metrics computed" if ss.model_results else "Pending evaluation",
        7: "Bias diagnostics available" if ss.models else "Pending bias check",
        8: "Segments assigned" if ss.recommendations_df is not None else "Pending recommendations",
        9: "Executive story ready" if 9 in ss.steps_done else "Pending export",
    }
    for num, icon, label in STEPS:
        status = "✅ Complete" if num in ss.steps_done else "○ Pending"
        journey.append({
            "Step": f"{num} {icon} {label}",
            "Status": status,
            "Notes": step_notes.get(num, ""),
        })
    st.dataframe(pd.DataFrame(journey), use_container_width=True, hide_index=True)

    st.subheader("Data Lineage")
    lineage_rows = [
        ("Raw ingestion", ss.raw_df.shape if ss.raw_df is not None else "—"),
        ("After cleaning", ss.cleaned_df.shape if ss.cleaned_df is not None else "—"),
        ("Feature matrix (train)", ss.X_train.shape if ss.X_train is not None else "—"),
        ("Feature matrix (test)", ss.X_test.shape if ss.X_test is not None else "—"),
        ("Recommendations scored", ss.recommendations_df.shape if ss.recommendations_df is not None else "—"),
    ]
    st.dataframe(
        pd.DataFrame(lineage_rows, columns=["Stage", "Shape (rows × cols)"]),
        use_container_width=True,
        hide_index=True,
    )

    if churn_rate is not None:
        st.subheader("Churn Status Assessment")
        if churn_rate > 0.25:
            _warn("Status: **CRITICAL** — churn above 25% threshold. Immediate retention programme required.")
        elif churn_rate > 0.15:
            _warn("Status: **ELEVATED** — proactive retention recommended.")
        else:
            _ok("Status: Within acceptable range.")

    if ss.clean_log:
        st.subheader("Cleaning Highlights")
        for entry in ss.clean_log[:6]:
            _ok(entry)

    if ss.eda_insights:
        st.subheader("Key EDA Insights")
        for insight in ss.eda_insights:
            _insight(insight)

    if ss.model_results:
        st.subheader("Model Scorecard")
        summary = pd.DataFrame({
            name: {k: v for k, v in res.items() if k not in ("y_pred", "y_prob")}
            for name, res in ss.model_results.items()
        }).T.round(4)
        st.dataframe(summary, use_container_width=True)

        if ss.feature_importance is not None:
            st.subheader("Top 8 Churn Drivers (Random Forest)")
            fi = ss.feature_importance.head(8).reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

    if ss.recommendations_df is not None:
        st.subheader("Segment Snapshot")
        seg = ss.recommendations_df.groupby("segment").agg(
            subscribers=("churn_probability", "count"),
            avg_prob=("churn_probability", "mean"),
        ).reset_index()
        st.dataframe(seg.round(3), use_container_width=True, hide_index=True)
        _insight(
            f"**{int((ss.recommendations_df['churn_probability'] >= 0.5).sum()):,} subscribers** "
            "have a churn probability ≥ 50% and should be prioritised for retention campaigns immediately."
        )

    st.subheader("Documentation & Methodology")
    with st.expander("Pipeline Methodology Reference", expanded=False):
        st.markdown(
            """
**Step 1 — Data Ingestion**  
Upload CSV/Excel, load 2,000-row demo data, or download the 500-row sample CSV.  
42 telco fields spanning identity, subscription, usage, network KPIs, CX, billing, and loyalty.

**Step 2 — Exploratory Data Analysis**  
Descriptive statistics, target distribution (bar chart uses column name for colour),  
Pearson correlation heatmap, missing-value audit, numeric histograms, categorical churn rates.

**Step 3 — Data Cleaning**  
Duplicate removal, null-token normalisation (N/A, null, None), whitespace stripping,  
median imputation, numeric type coercion, re-imputation after coercion.

**Step 4 — Feature Engineering**  
Derived KPIs: `charge_per_mb`, `complaint_rate`, `engagement_score` (z-score mean only),  
`tenure_bucket`, `resolution_hours`, `sla_status`.  
Categorical encoding with `_SKIP_ENCODE_COLS` and `_MAX_OHE_CARDINALITY=30`.  
Binary target validation via `_prepare_binary_target`. Safe stratified split via `_safe_train_test_split`.  
Imputation and StandardScaler fit on train only.

**Step 5 — Model Training**  
Six classifiers: LogisticRegression, RandomForest, GradientBoosting, SVC(probability=True),  
KNeighborsClassifier, AdaBoostClassifier. NaN safety impute before train.

**Step 6 — Model Evaluation**  
Confusion matrix, ROC curves for all models, classification report, probability histogram.

**Step 7 — Bias Detection**  
Learning curves and train vs test AUC gap per model (overfit / underfit / good fit).

**Step 8 — Recommendations**  
Churn probability scoring, 7 segments, product + campaign mapping, CSV export.

**Step 9 — Data Story**  
Executive UI, `_generate_pptx` (executive deck), `_generate_technical_pptx` (technical deck), JSON export.
"""
        )

    st.subheader("Export Centre")
    st.markdown(
        "**Business audience:** KPIs, churn profile, scorecard, campaigns, action plan.  \n"
        "**Technical audience:** pipeline architecture, feature engineering, model zoo, bias diagnostics."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        try:
            ppt_bytes = _generate_pptx()
            st.download_button(
                "📊 Download Executive Deck (.pptx)",
                data=ppt_bytes,
                file_name=f"telco_executive_{datetime.now().strftime('%Y%m%d')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Executive PPT generation failed: {exc}")

    with col_b:
        try:
            tech_bytes = _generate_technical_pptx()
            st.download_button(
                "🔬 Download Technical Deck (.pptx)",
                data=tech_bytes,
                file_name=f"telco_technical_{datetime.now().strftime('%Y%m%d')}.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Technical PPT generation failed: {exc}")

    with col_c:
        export_payload = {
            "generated_at": datetime.now().isoformat(),
            "dataset": ss.dataset_name,
            "target_column": ss.target_col,
            "rows": int(df.shape[0]) if df is not None else 0,
            "features_used": ss.feature_cols,
            "steps_completed": sorted(ss.steps_done),
            "eda_insights": ss.eda_insights,
            "clean_log": ss.clean_log,
            "model_results": {
                name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in res.items() if k not in ("y_pred", "y_prob")}
                for name, res in ss.model_results.items()
            } if ss.model_results else {},
            "best_model": best,
            "feature_importance_top10": (
                ss.feature_importance.head(10).to_dict() if ss.feature_importance is not None else {}
            ),
            "segment_summary": (
                ss.recommendations_df.groupby("segment")["churn_probability"].agg(["count", "mean"]).to_dict()
                if ss.recommendations_df is not None else {}
            ),
        }
        st.download_button(
            "📄 Download JSON Report",
            data=json.dumps(export_payload, indent=2, default=str).encode(),
            file_name=f"telco_pipeline_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True,
        )

    if st.button("Mark Pipeline Complete ✅", type="primary"):
        _complete_step(9)
        st.balloons()
        st.success("🎉 Telco Data Science Pipeline complete!")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
STEP_FNS = {
    1: step_ingest,
    2: step_eda,
    3: step_clean,
    4: step_features,
    5: step_train,
    6: step_evaluate,
    7: step_bias,
    8: step_recommendations,
    9: step_story,
}

_nav_sidebar()
STEP_FNS[ss.current_step if ss.current_step in STEP_FNS else len(STEPS)]()
