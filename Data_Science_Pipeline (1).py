"""
Telco Data Science Pipeline  —  Streamlit Web Application
==========================================================
End-to-end data science roadmap with a step-by-step UI:

  Step 1  Data Ingestion          Upload CSV / use demo data
  Step 2  Exploratory Analysis    Stats, distributions, correlations
  Step 3  Data Cleaning           Nulls, duplicates, type coercion
  Step 4  Feature Engineering     Encoding, scaling, derived KPIs
  Step 5  Model Training          Logistic Regression + Random Forest
  Step 6  Model Evaluation        Accuracy, ROC-AUC, F1, confusion matrix
  Step 7  Bias Detection          Learning curves – overfit / underfit
  Step 8  Recommendations         Product & campaign segmentation
  Step 9  Data Story              Presentation-ready insight report

Run:
    streamlit run "ETL Pipeline/Data_Science_Pipeline.py"
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
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco DS Pipeline",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
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
# STEP DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
STEPS = [
    (1,  "📥",  "Data Ingestion"),
    (2,  "🔍",  "Exploratory Analysis"),
    (3,  "🧹",  "Data Cleaning"),
    (4,  "⚙️",  "Feature Engineering"),
    (5,  "🤖",  "Model Training"),
    (6,  "📊",  "Model Evaluation"),
    (7,  "📉",  "Bias Detection"),
    (8,  "💡",  "Recommendations"),
    (9,  "🎯",  "Data Story"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_STATE_DEFAULTS: dict = {
    "current_step":     1,
    "raw_df":           None,
    "cleaned_df":       None,
    "featured_df":      None,
    "target_col":       None,
    "feature_cols":     [],
    "X_train":          None,
    "X_test":           None,
    "y_train":          None,
    "y_test":           None,
    "models":           {},
    "model_results":    {},
    "feature_importance": None,
    "recommendations_df": None,
    "steps_done":       set(),
    "eda_insights":     [],
    "clean_log":        [],
    "dataset_name":     "demo",
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

ss = st.session_state  # shorthand

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def _nav_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📡 Telco DS Pipeline")
        st.caption("End-to-end Data Science Roadmap")
        st.divider()

        max_unlocked = max(ss.steps_done, default=0) + 1

        for num, icon, label in STEPS:
            if num in ss.steps_done:
                prefix = "✅"
            elif num == ss.current_step:
                prefix = "🔵"
            else:
                prefix = f"○"

            disabled = num > max_unlocked
            if st.button(
                f"{prefix} {icon} {label}",
                key=f"nav_{num}",
                width='stretch',
                disabled=disabled,
            ):
                ss.current_step = num
                st.rerun()

        st.divider()
        # Progress bar
        pct = len(ss.steps_done) / len(STEPS)
        st.progress(pct, text=f"Progress: {int(pct*100)}%")

        if ss.raw_df is not None:
            st.caption(
                f"📋 **{ss.dataset_name}**  \n"
                f"{ss.raw_df.shape[0]:,} rows × {ss.raw_df.shape[1]} cols"
            )
        if ss.target_col:
            st.caption(f"🎯 Target: `{ss.target_col}`")

        st.divider()
        if st.button("🔄 Reset Pipeline", width='stretch'):
            for k, v in _STATE_DEFAULTS.items():
                st.session_state[k] = v if not isinstance(v, set) else set()
            st.rerun()


_nav_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _step_header(num: int, icon: str, title: str) -> None:
    st.markdown(
        f'<div class="step-header"><span style="color:#2196f3;">Step {num}</span>'
        f' {icon} {title}</div>',
        unsafe_allow_html=True,
    )


def _complete_step(num: int) -> None:
    ss.steps_done.add(num)
    ss.current_step = num + 1


def _kpi(col, value, label, variant: str = "") -> None:
    col.markdown(
        f'<div class="kpi-card {variant}">'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _insight(msg: str) -> None:
    st.markdown(f'<div class="insight-box">💡 {msg}</div>', unsafe_allow_html=True)


def _warn(msg: str) -> None:
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def _ok(msg: str) -> None:
    st.markdown(f'<div class="ok-box">✔ {msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def _demo_data(n: int = 2_000) -> pd.DataFrame:
    """
    Full-featured synthetic telco dataset with 42 columns covering:
    Identity, Subscription, Usage, Network KPIs, Customer Experience,
    Billing, Loyalty — plus injected nulls, bad tokens & duplicates
    to exercise every pipeline step.
    """
    from datetime import timedelta  # also available at module level

    rng      = np.random.RandomState(42)
    plans    = ["Prepaid Basic", "Prepaid Smart", "Postpaid 199", "Postpaid 499",
                "Postpaid 999", "IoT SIM", "Business Pro"]
    regs     = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
                "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape"]
    devs     = ["Smartphone", "Feature Phone", "IoT Device", "Tablet", "Router", "Wearable"]
    networks = ["4G LTE", "3G", "5G", "2G", "NB-IoT"]
    channels = ["Direct Sales", "Online Portal", "Retail Store", "Agent", "Referral"]
    contract = ["Month-to-Month", "One Year", "Two Year"]
    payment  = ["Credit Card", "Debit Order", "EFT", "Voucher", "Mobile Money"]
    svc_stat = ["Active", "Suspended", "Terminated", "Porting Out"]
    nodes    = ["BTS_JHB_001", "BTS_CPT_007", "BTS_DBN_003",
                "BTS_PTA_012", "BTS_EL_005", "BTS_NM_009"]

    tenure          = rng.randint(1, 84, n)
    charges         = rng.uniform(99, 1299, n).round(2)
    data_mb         = np.maximum(0, rng.exponential(2_000, n)).round(1)
    data_overage_mb = np.maximum(0, rng.exponential(300, n)).round(1)
    calls_min       = np.maximum(0, rng.exponential(280, n)).round(1)
    sms             = rng.poisson(30, n).astype(int)
    comps           = rng.poisson(1.4, n).astype(int)
    refunds         = np.where(comps > 0, rng.uniform(0, 500, n).round(2), 0.0)
    latency         = rng.uniform(10, 450, n).round(1)
    signal_dbm      = rng.uniform(-110, -50, n).round(1)
    roaming         = rng.choice([0, 1], n, p=[0.82, 0.18])
    intl_calls      = np.where(roaming, rng.uniform(0, 120, n).round(2), 0.0)
    upgrades        = rng.poisson(0.5, n).astype(int)
    reactivations   = rng.poisson(0.2, n).astype(int)
    nps_score       = rng.randint(0, 11, n)
    csat_score      = rng.uniform(1, 5, n).round(1)
    avg_recharge    = np.where(rng.rand(n) < 0.5, rng.uniform(20, 150, n).round(2), 0.0)
    bill_shock      = ((charges > 800) | (data_overage_mb > 1_000)).astype(int)
    pay_fail        = rng.poisson(0.3, n).astype(int)
    days_overdue    = np.where(pay_fail > 0, rng.randint(1, 60, n), 0)
    support_calls   = rng.poisson(1.1, n).astype(int)
    app_logins      = rng.poisson(12, n).astype(int)
    paperless       = rng.choice([0, 1], n)
    last_recharge   = rng.randint(0, 90, n)
    sla_hours       = rng.choice([4, 8, 24, 48], n)

    base = datetime(2024, 1, 1)
    act_dates   = [(base + timedelta(days=int(d))).strftime("%Y-%m-%d")
                   for d in rng.randint(0, 730, n)]
    created_dt  = [(base + timedelta(days=int(d), hours=int(h))).strftime("%Y-%m-%dT%H:%M:%S")
                   for d, h in zip(rng.randint(0, 365, n), rng.randint(0, 24, n))]
    resolve_hrs = rng.uniform(0.5, 96, n).round(1)
    closed_dt   = [(datetime.strptime(c, "%Y-%m-%dT%H:%M:%S") + timedelta(hours=float(r)))
                   .strftime("%Y-%m-%dT%H:%M:%S")
                   for c, r in zip(created_dt, resolve_hrs)]
    id_numbers  = [
        f"{rng.randint(800101,991231):06d}{rng.randint(1000,9999):04d}"
        f"{rng.randint(0,2)}{rng.randint(80,99):02d}"
        for _ in range(n)
    ]

    churn_p = (
        0.06
        + 0.30 * (tenure < 6).astype(float)
        + 0.22 * (comps > 3).astype(float)
        + 0.18 * (data_mb < 150).astype(float)
        + 0.12 * (charges > 900).astype(float)
        + 0.10 * (nps_score < 4).astype(float)
        + 0.08 * (pay_fail > 0).astype(float)
        - 0.12 * (tenure > 36).astype(float)
        - 0.08 * (upgrades > 0).astype(float)
    ).clip(0.02, 0.95)
    churn = (rng.rand(n) < churn_p).astype(int)

    # Build charges as object so we can inject bad tokens
    charges_str = [str(v) for v in charges]
    for idx in rng.randint(0, n, 6):
        charges_str[idx] = "-"

    df = pd.DataFrame({
        # ── Identity ──────────────────────────────────────────
        "subscriber_id":          [f"SUB{i:05d}" for i in range(n)],
        "account_number":         [f"ACC{rng.randint(100000,999999)}" for _ in range(n)],
        "msisdn":                 [f"2760{rng.randint(1_000_000,9_999_999)}" for _ in range(n)],
        "id_number":              id_numbers,
        # ── Subscription ──────────────────────────────────────
        "plan":                   rng.choice(plans, n),
        "contract_type":          rng.choice(contract, n),
        "network_type":           rng.choice(networks, n),
        "service_status":         rng.choice(svc_stat, n, p=[0.80, 0.08, 0.07, 0.05]),
        "acquisition_channel":    rng.choice(channels, n),
        "payment_method":         rng.choice(payment, n),
        "region":                 rng.choice(regs, n),
        "device_type":            rng.choice(devs, n),
        "node_id":                rng.choice(nodes, n),
        # ── Tenure & dates ────────────────────────────────────
        "tenure_months":          tenure,
        "activation_date":        act_dates,
        "created_date":           created_dt,
        "closed_date":            closed_dt,
        "last_recharge_days_ago": last_recharge,
        "sla_hours":              sla_hours,
        # ── Usage ─────────────────────────────────────────────
        "monthly_charges":        charges_str,   # object; contains "-" tokens
        "data_volume_mb":         data_mb,
        "data_overage_mb":        data_overage_mb,
        "call_minutes":           calls_min,
        "sms_count":              sms,
        "intl_call_minutes":      intl_calls,
        "roaming_flag":           roaming,
        "avg_recharge_amount":    avg_recharge,
        # ── Network KPIs ──────────────────────────────────────
        "latency_ms":             latency,
        "signal_strength_dbm":    signal_dbm,
        # ── Customer Experience ───────────────────────────────
        "complaint_count":        comps,
        "support_calls":          support_calls,
        "nps_score":              nps_score,
        "csat_score":             csat_score,
        "app_logins_monthly":     app_logins,
        # ── Billing ───────────────────────────────────────────
        "refund_amount":          refunds,
        "payment_failures":       pay_fail,
        "days_overdue":           days_overdue,
        "bill_shock_flag":        bill_shock,
        "paperless_billing":      paperless,
        # ── Loyalty ───────────────────────────────────────────
        "plan_upgrades":          upgrades,
        "reactivation_count":     reactivations,
        # ── Target ────────────────────────────────────────────
        "churn":                  churn,
    })

    # Inject ~4% NaN in key numeric cols  -> tests imputation
    for col in ["data_volume_mb", "call_minutes", "refund_amount",
                "latency_ms", "csat_score", "signal_strength_dbm"]:
        df.loc[rng.rand(n) < 0.04, col] = np.nan

    # Inject ~2% blank strings in categoricals  -> tests whitespace/null handling
    for col in ["plan", "region", "device_type", "acquisition_channel"]:
        df.loc[rng.rand(n) < 0.02, col] = ""

    # Inject bad null tokens  -> tests null-token normalisation
    df.loc[rng.randint(0, n, 8), "payment_method"] = "N/A"
    df.loc[rng.randint(0, n, 5), "contract_type"]  = "null"
    df.loc[rng.randint(0, n, 4), "service_status"] = "None"

    # Inject ~1.5% exact duplicate rows  -> tests deduplication
    df = pd.concat([df, df.sample(30, random_state=42)], ignore_index=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────
def step_ingest() -> None:
    _step_header(1, "📥", "Data Ingestion")

    st.markdown(
        "Upload your telecom dataset (CSV / Excel), download the ready-made "
        "sample CSV to test all checklist items, or click **Use Demo Data** "
        "to load a synthetic 2,000-subscriber dataset directly."
    )

    # ── Field dictionary expander ────────────────────────────────────────────
    with st.expander("📋  Full Field Dictionary — 42 Telco Columns", expanded=False):
        field_dict = pd.DataFrame([
            # Identity
            ("subscriber_id",          "Identity",      "Text",    "Unique subscriber key  e.g. SUB00001"),
            ("account_number",         "Identity",      "Text",    "Billing account reference"),
            ("msisdn",                 "Identity",      "Text",    "Mobile number in E.164 format (2760xxxxxxx)"),
            ("id_number",              "Identity",      "Text",    "South African ID number (13 digits)"),
            # Subscription
            ("plan",                   "Subscription",  "Text",    "Price plan name"),
            ("contract_type",          "Subscription",  "Text",    "Month-to-Month / One Year / Two Year"),
            ("network_type",           "Subscription",  "Text",    "2G / 3G / 4G LTE / 5G / NB-IoT"),
            ("service_status",         "Subscription",  "Text",    "Active / Suspended / Terminated / Porting Out"),
            ("acquisition_channel",    "Subscription",  "Text",    "How subscriber was acquired"),
            ("payment_method",         "Subscription",  "Text",    "Credit Card / Debit Order / EFT / Voucher / Mobile Money"),
            ("region",                 "Subscription",  "Text",    "Province"),
            ("device_type",            "Subscription",  "Text",    "Smartphone / Feature Phone / IoT Device etc."),
            ("node_id",                "Network",       "Text",    "Base transceiver station ID"),
            # Tenure & dates
            ("tenure_months",          "Tenure",        "Integer", "Months since activation"),
            ("activation_date",        "Tenure",        "Date",    "Service activation date  YYYY-MM-DD"),
            ("created_date",           "Dates",         "DateTime","Ticket / transaction created  ISO 8601"),
            ("closed_date",            "Dates",         "DateTime","Ticket / transaction closed  ISO 8601"),
            ("last_recharge_days_ago", "Tenure",        "Integer", "Days since last recharge event"),
            ("sla_hours",              "SLA",           "Integer", "SLA target in hours  4/8/24/48"),
            # Usage
            ("monthly_charges",        "Billing",       "Float",   "Monthly invoice amount (ZAR) — may contain '-' tokens"),
            ("data_volume_mb",         "Usage",         "Float",   "Total data consumed in MB — ~4% nulls injected"),
            ("data_overage_mb",        "Usage",         "Float",   "Data consumed above plan bundle"),
            ("call_minutes",           "Usage",         "Float",   "Voice call minutes — ~4% nulls injected"),
            ("sms_count",              "Usage",         "Integer", "SMS messages sent"),
            ("intl_call_minutes",      "Usage",         "Float",   "International call minutes (0 if not roaming)"),
            ("roaming_flag",           "Usage",         "Binary",  "1 = subscriber roamed in the period"),
            ("avg_recharge_amount",    "Usage",         "Float",   "Average top-up amount (0 for postpaid)"),
            # Network KPIs
            ("latency_ms",             "Network KPI",   "Float",   "Average round-trip latency (ms)"),
            ("signal_strength_dbm",    "Network KPI",   "Float",   "Average signal strength (dBm, -110 to -50)"),
            # Customer Experience
            ("complaint_count",        "CX",            "Integer", "Complaints logged in period"),
            ("support_calls",          "CX",            "Integer", "Calls to support centre"),
            ("nps_score",              "CX",            "Integer", "Net Promoter Score  0-10"),
            ("csat_score",             "CX",            "Float",   "Customer Satisfaction score  1.0-5.0"),
            ("app_logins_monthly",     "CX",            "Integer", "Self-service app logins per month"),
            # Billing
            ("refund_amount",          "Billing",       "Float",   "Refunds issued (ZAR)"),
            ("payment_failures",       "Billing",       "Integer", "Failed payment attempts"),
            ("days_overdue",           "Billing",       "Integer", "Days payment is overdue"),
            ("bill_shock_flag",        "Billing",       "Binary",  "1 = bill > R800 or data overage > 1 GB"),
            ("paperless_billing",      "Billing",       "Binary",  "1 = enrolled in e-billing"),
            # Loyalty
            ("plan_upgrades",          "Loyalty",       "Integer", "Number of plan upgrades"),
            ("reactivation_count",     "Loyalty",       "Integer", "Times reactivated after suspension"),
            # Target
            ("churn",                  "TARGET",        "Binary",  "1 = churned in period  (predict this)"),
        ], columns=["Column", "Category", "Type", "Description"])
        st.dataframe(field_dict, width="stretch", hide_index=True, height=550)

    # ── Checklist ────────────────────────────────────────────────────────────
    with st.expander("✅  Pipeline Checklist — What each field tests", expanded=False):
        checklist = pd.DataFrame([
            ("Step 1 Ingestion",       "All 42 columns",           "CSV upload / demo data load"),
            ("Step 2 EDA",             "All numeric columns",      "Distributions, correlation heatmap, class balance"),
            ("Step 2 EDA",             "plan, region, device_type","Categorical bar + churn rate charts"),
            ("Step 2 EDA",             "data_volume_mb, latency_ms","Missing value map (4% injected)"),
            ("Step 3 Cleaning",        "monthly_charges",          "Auto-coerce '-' tokens to numeric"),
            ("Step 3 Cleaning",        "N/A, null, None tokens",   "Null-token normalisation (payment_method, contract_type, service_status)"),
            ("Step 3 Cleaning",        "Blank strings",            "Whitespace strip + blank -> null (plan, region, device_type, channel)"),
            ("Step 3 Cleaning",        "10 duplicate rows",        "Duplicate row removal"),
            ("Step 3 Cleaning",        "6 numeric cols",           "Median/mean imputation of NaN values"),
            ("Step 4 Engineering",     "monthly_charges, data_volume_mb", "Derived: charge_per_mb"),
            ("Step 4 Engineering",     "complaint_count, tenure_months",  "Derived: complaint_rate"),
            ("Step 4 Engineering",     "call_minutes, sms_count, data_volume_mb", "Derived: engagement_score"),
            ("Step 4 Engineering",     "tenure_months",            "Derived: tenure_bucket (binned)"),
            ("Step 4 Engineering",     "created_date, closed_date, sla_hours", "SLA status + resolution_hours"),
            ("Step 4 Engineering",     "plan, region, device_type etc.", "One-hot encoding of categoricals"),
            ("Step 5 Training",        "churn (target)",           "Logistic Regression + Random Forest + Gradient Boosting"),
            ("Step 5 Training",        "All numeric features",     "Feature importance (Random Forest)"),
            ("Step 6 Evaluation",      "churn",                    "Confusion matrix, ROC-AUC, F1, precision, recall"),
            ("Step 6 Evaluation",      "churn_probability",        "Probability distribution histogram"),
            ("Step 7 Bias Detection",  "All features",             "Learning curves: overfit / underfit / good fit"),
            ("Step 7 Bias Detection",  "Train vs Test",            "Train/Test AUC gap per model"),
            ("Step 8 Recommendations", "churn_probability",        "7 risk segments with product + campaign mapping"),
            ("Step 9 Data Story",      "All stages",               "Executive presentation: churn donut, drivers, scorecard, action plan"),
        ], columns=["Step", "Field(s) Used", "What is Tested"])
        st.dataframe(checklist, width="stretch", hide_index=True, height=480)

    # ── Load buttons ─────────────────────────────────────────────────────────
    col_upload, col_demo, col_dl = st.columns([3, 1, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )
    with col_demo:
        use_demo = st.button("Use Demo Data", width="stretch", type="primary")
    with col_dl:
        # Provide the pre-generated sample CSV as a download
        sample_path = Path(__file__).parent / "telco_sample_data.csv"
        if sample_path.exists():
            st.download_button(
                label="Download Sample CSV",
                data=sample_path.read_bytes(),
                file_name="telco_sample_data.csv",
                mime="text/csv",
                help="500-row CSV with all 42 fields — upload it back to test every step",
            )
        else:
            st.caption("Sample CSV not found")

    if use_demo:
        ss.raw_df       = _demo_data()
        ss.dataset_name = "Telco Demo Dataset (2,000 rows × 42 cols)"
        st.success(f"Demo dataset loaded: {ss.raw_df.shape[0]:,} rows × {ss.raw_df.shape[1]} cols")

    if uploaded is not None:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded, low_memory=False)
            ss.raw_df       = df
            ss.dataset_name = uploaded.name
            st.success(f"Loaded **{uploaded.name}**: {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    if ss.raw_df is None:
        st.info("Load data above to continue.")
        return

    df = ss.raw_df

    # Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(20), width='stretch', height=280)

    # Quick stats row
    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, f"{df.shape[0]:,}",   "Total Records")
    _kpi(c2, f"{df.shape[1]}",     "Columns",        "green")
    _kpi(c3, f"{df.isnull().sum().sum():,}", "Missing Cells", "orange")
    _kpi(c4, f"{df.duplicated().sum():,}",  "Duplicates",    "purple")

    # Column inventory
    st.subheader("Column Inventory")
    inv = pd.DataFrame({
        "Column":   df.columns,
        "Type":     df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Null %":   (df.isnull().mean() * 100).round(1).values,
        "Unique":   [df[c].nunique() for c in df.columns],
        "Sample":   [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "" for c in df.columns],
    })
    st.dataframe(inv, width='stretch', hide_index=True)

    # Target column selection
    st.subheader("Target Column (for Churn Prediction)")
    binary_cols = [c for c in df.columns if df[c].nunique() == 2]
    churn_guess = next(
        (c for c in df.columns if "churn" in c.lower()), None
    ) or (binary_cols[0] if binary_cols else df.columns[-1])

    ss.target_col = st.selectbox(
        "Select the churn / target column (must be 0/1 or binary):",
        options=df.columns.tolist(),
        index=list(df.columns).index(churn_guess) if churn_guess in df.columns else 0,
    )

    # Auto-encode target if it's not numeric
    target_series = df[ss.target_col].dropna()
    if target_series.dtype == object:
        unique_vals = sorted(target_series.unique())
        _warn(
            f"Target `{ss.target_col}` is text. "
            f"Values: {unique_vals[:5]}. "
            "It will be label-encoded in the Feature Engineering step."
        )

    if st.button("Proceed to Exploratory Analysis →", type="primary"):
        _complete_step(1)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
def step_eda() -> None:
    _step_header(2, "🔍", "Exploratory Data Analysis")
    df  = ss.raw_df
    tgt = ss.target_col
    ss.eda_insights = []

    # ── Numeric summary ──────────────────────────────────────────────────────
    st.subheader("Descriptive Statistics")
    num_df = df.select_dtypes(include="number")
    st.dataframe(num_df.describe().T.round(2), width='stretch')

    # ── Target distribution ──────────────────────────────────────────────────
    st.subheader(f"Target Distribution  —  `{tgt}`")
    vc = df[tgt].value_counts(dropna=False).reset_index()
    vc.columns = [tgt, "count"]
    vc["pct"] = (vc["count"] / vc["count"].sum() * 100).round(1)

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig = px.pie(vc, names=tgt, values="count",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title="Class Distribution")
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')
    with col_bar:
        fig2 = px.bar(vc, x=tgt, y="count", text="pct",
                      color=tgt.astype(str) if vc.shape[0] < 10 else None,
                      title="Class Counts")
        fig2.update_traces(texttemplate="%{text}%", textposition="outside")
        fig2.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig2, width='stretch')

    if len(vc) == 2:
        min_pct = vc["pct"].min()
        if min_pct < 20:
            _warn(f"Class imbalance detected: minority class = {min_pct:.1f}%. "
                  "Consider resampling or using class_weight='balanced' in models.")
            ss.eda_insights.append(f"Class imbalance: minority = {min_pct:.1f}%")
        else:
            _ok("Class distribution is reasonably balanced.")

    # ── Numeric distributions ────────────────────────────────────────────────
    num_cols = [c for c in num_df.columns if c != tgt and num_df[c].nunique() > 5]
    if num_cols:
        st.subheader("Numeric Feature Distributions")
        cols_pp = min(3, len(num_cols))
        rows_pp = (len(num_cols) + cols_pp - 1) // cols_pp
        fig = make_subplots(rows=rows_pp, cols=cols_pp,
                            subplot_titles=num_cols[:rows_pp * cols_pp])
        for idx, col in enumerate(num_cols[:rows_pp * cols_pp]):
            r, c = divmod(idx, cols_pp)
            fig.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, nbinsx=30,
                             showlegend=False,
                             marker_color="#42a5f5"),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(height=280 * rows_pp, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.subheader("Correlation Heatmap")
        corr = df[num_cols + ([tgt] if tgt in num_cols else [])].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}",
            colorbar=dict(title="r"),
        ))
        fig_corr.update_layout(height=500, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_corr, width='stretch')

        # Top correlations with target
        if tgt in corr.columns:
            top = (corr[tgt].drop(tgt).abs()
                   .sort_values(ascending=False)
                   .head(5))
            _insight(
                f"Top features correlated with **{tgt}**: "
                + ", ".join([f"`{k}` ({v:.2f})" for k, v in top.items()])
            )
            ss.eda_insights.append(f"Top correlators with {tgt}: {', '.join(top.index)}")

    # ── Categorical features ──────────────────────────────────────────────────
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != tgt
                and df[c].nunique() <= 20]
    if cat_cols:
        st.subheader("Categorical Feature Analysis")
        chosen = st.selectbox("Select categorical column:", cat_cols)
        churn_by_cat = (
            df.groupby(chosen)[tgt]
            .mean()
            .reset_index()
            .rename(columns={tgt: "churn_rate"})
            .sort_values("churn_rate", ascending=False)
        ) if pd.api.types.is_numeric_dtype(df[tgt]) else None

        fig_cat = px.histogram(df, x=chosen, color=str(tgt),
                               barmode="group",
                               title=f"{chosen} by {tgt}",
                               color_discrete_sequence=px.colors.qualitative.Safe)
        fig_cat.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_cat, width='stretch')

        if churn_by_cat is not None:
            fig_rate = px.bar(churn_by_cat, x=chosen, y="churn_rate",
                              color="churn_rate",
                              color_continuous_scale="Reds",
                              title=f"Churn Rate by {chosen}")
            fig_rate.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_rate, width='stretch')

    # ── Missing value map ─────────────────────────────────────────────────────
    null_pct = df.isnull().mean() * 100
    null_cols = null_pct[null_pct > 0]
    if len(null_cols) > 0:
        st.subheader("Missing Value Summary")
        fig_null = px.bar(
            x=null_cols.index, y=null_cols.values,
            labels={"x": "Column", "y": "Missing %"},
            color=null_cols.values, color_continuous_scale="Oranges",
            title="% Missing per Column",
        )
        fig_null.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_null, width='stretch')
        high_null = null_cols[null_cols > 30]
        if not high_null.empty:
            _warn(f"Columns with >30% missing: {', '.join(high_null.index)} — consider dropping.")
            ss.eda_insights.append(f"High nulls (>30%): {', '.join(high_null.index)}")

    if st.button("Proceed to Data Cleaning →", type="primary"):
        _complete_step(2)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def step_clean() -> None:
    _step_header(3, "🧹", "Data Cleaning & Preprocessing")
    df  = ss.raw_df.copy()
    log = []

    st.markdown("Configure cleaning rules then click **Apply Cleaning**.")

    with st.expander("Cleaning Options", expanded=True):
        col_a, col_b = st.columns(2)
        drop_dups   = col_a.checkbox("Remove duplicate rows",         value=True)
        drop_nullpct= col_a.number_input("Drop column if >% null",    value=50, min_value=0, max_value=100)
        fill_num    = col_b.selectbox("Fill numeric nulls with:",     ["median", "mean", "zero"])
        fill_cat    = col_b.selectbox("Fill text nulls with:",        ["mode", "UNKNOWN"])
        strip_ws    = col_a.checkbox("Strip whitespace from strings", value=True)
        fix_types   = col_b.checkbox("Auto-coerce obvious types",     value=True)

    if st.button("Apply Cleaning", type="primary"):
        before_rows = len(df)
        before_cols = len(df.columns)

        # 1. Strip whitespace
        if strip_ws:
            str_cols = df.select_dtypes("object").columns
            for c in str_cols:
                df[c] = df[c].str.strip()
            log.append(f"Stripped whitespace from {len(str_cols)} text columns.")

        # 2. Drop columns with too many nulls
        null_pct = df.isnull().mean() * 100
        to_drop  = null_pct[null_pct > drop_nullpct].index.tolist()
        if to_drop:
            df = df.drop(columns=to_drop)
            log.append(f"Dropped {len(to_drop)} high-null columns: {to_drop}")

        # 3. Remove duplicates
        if drop_dups:
            before_d = len(df)
            df = df.drop_duplicates()
            removed = before_d - len(df)
            log.append(f"Removed {removed:,} duplicate rows.")

        # 4. Fill nulls — numeric
        num_cols = df.select_dtypes(include="number").columns
        imp_num  = SimpleImputer(
            strategy="median" if fill_num == "median" else
                      "mean"   if fill_num == "mean"   else "constant",
            fill_value=0 if fill_num == "zero" else None,
        )
        if len(num_cols) > 0:
            df[num_cols] = imp_num.fit_transform(df[num_cols])
            log.append(f"Imputed {len(num_cols)} numeric columns with {fill_num}.")

        # 5. Fill nulls — categorical
        cat_cols = df.select_dtypes("object").columns
        for c in cat_cols:
            if df[c].isnull().any():
                fill_val = df[c].mode().iloc[0] if fill_cat == "mode" else "UNKNOWN"
                df[c] = df[c].fillna(fill_val)
        if len(cat_cols) > 0:
            log.append(f"Imputed {len(cat_cols)} text columns with {fill_cat}.")

        # 6. Auto type coercion
        if fix_types:
            for c in df.select_dtypes("object").columns:
                if c == ss.target_col:
                    continue
                try:
                    converted = pd.to_numeric(df[c], errors="coerce")
                    if converted.notnull().mean() > 0.9:
                        df[c] = converted
                        log.append(f"Auto-coerced `{c}` to numeric.")
                except Exception:
                    pass

        after_rows = len(df)
        after_cols = len(df.columns)
        log.append(
            f"Final shape: {after_rows:,} rows × {after_cols} cols "
            f"(removed {before_rows - after_rows:,} rows, {before_cols - after_cols} cols)."
        )

        ss.cleaned_df = df
        ss.clean_log  = log
        st.success("Cleaning complete!")

    if ss.cleaned_df is not None:
        df = ss.cleaned_df
        st.subheader("Cleaning Log")
        for entry in ss.clean_log:
            _ok(entry)

        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df.head(20), width='stretch', height=260)

        c1, c2, c3, c4 = st.columns(4)
        _kpi(c1, f"{df.shape[0]:,}",             "Rows After",  "green")
        _kpi(c2, f"{df.shape[1]}",               "Columns",     "green")
        _kpi(c3, f"{df.isnull().sum().sum()}",   "Remaining Nulls", "orange")
        _kpi(c4, f"{df.duplicated().sum()}",     "Duplicates",  "green")

        if st.button("Proceed to Feature Engineering →", type="primary"):
            _complete_step(3)
            st.rerun()
    else:
        st.info("Apply cleaning above to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def step_features() -> None:
    _step_header(4, "⚙️", "Feature Engineering")
    df  = ss.cleaned_df.copy()
    tgt = ss.target_col

    st.markdown(
        "Derived features, encoding, and scaling are applied automatically. "
        "Select which columns to include as model features."
    )

    # ── Derived telco features ───────────────────────────────────────────────
    st.subheader("Derived KPI Features")
    derived = []

    # Charge per MB
    if "monthly_charges" in df.columns and "data_volume_mb" in df.columns:
        df["charge_per_mb"] = (df["monthly_charges"] / df["data_volume_mb"].replace(0, np.nan)).round(4)
        df["charge_per_mb"] = df["charge_per_mb"].fillna(0)
        derived.append("charge_per_mb  = monthly_charges / data_volume_mb")

    # Complaint ratio
    if "complaint_count" in df.columns and "tenure_months" in df.columns:
        df["complaint_rate"] = (df["complaint_count"] / df["tenure_months"].replace(0, 1)).round(4)
        derived.append("complaint_rate = complaint_count / tenure_months")

    # Engagement score (normalised combo)
    eng_cols = [c for c in ["call_minutes", "sms_count", "data_volume_mb"] if c in df.columns]
    if eng_cols:
        df["engagement_score"] = df[eng_cols].apply(
            lambda r: sum(r / r.max() if r.max() > 0 else 0 for _ in [1]), axis=1
        )
        # Simpler: sum of z-scores
        for c in eng_cols:
            std = df[c].std()
            mean = df[c].mean()
            df[f"{c}_z"] = (df[c] - mean) / (std if std > 0 else 1)
        df["engagement_score"] = df[[f"{c}_z" for c in eng_cols]].mean(axis=1).round(4)
        df = df.drop(columns=[f"{c}_z" for c in eng_cols])
        derived.append("engagement_score = mean z-score of call/SMS/data usage")

    # Tenure bucket
    if "tenure_months" in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure_months"],
            bins=[0, 6, 12, 24, 48, 999],
            labels=["0-6m", "6-12m", "12-24m", "24-48m", "48m+"],
        ).astype(str)
        derived.append("tenure_bucket  = binned tenure_months")

    for d in derived:
        _ok(d)

    # ── Encode categoricals ───────────────────────────────────────────────────
    st.subheader("Categorical Encoding")
    cat_cols = df.select_dtypes("object").columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ("subscriber_id", "msisdn")]

    if cat_cols:
        for c in cat_cols:
            if df[c].nunique() <= 2 or c == tgt:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                _ok(f"Label-encoded `{c}` ({df[c].nunique()} classes)")
            else:
                dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
                df      = pd.concat([df.drop(columns=[c]), dummies], axis=1)
                _ok(f"One-hot-encoded `{c}` → {len(dummies.columns)} dummy columns")

    # ── Feature selection ─────────────────────────────────────────────────────
    st.subheader("Select Model Features")
    exclude = {"subscriber_id", "msisdn", tgt}
    all_features = [c for c in df.columns if c not in exclude
                    and pd.api.types.is_numeric_dtype(df[c])]

    selected = st.multiselect(
        "Features to include:",
        options=all_features,
        default=all_features[:min(len(all_features), 15)],
    )

    if not selected:
        st.warning("Select at least one feature.")
        return

    ss.feature_cols = selected

    # ── Scale ────────────────────────────────────────────────────────────────
    st.subheader("Feature Scaling")
    scale_method = st.selectbox("Scaling method:", ["StandardScaler", "None"])

    if st.button("Apply Features & Split Data", type="primary"):
        feat_df = df[selected].copy()
        target  = df[tgt].copy()

        # Encode target if needed
        if target.dtype == object:
            le_tgt = LabelEncoder()
            target = pd.Series(le_tgt.fit_transform(target), name=tgt)

        # Scale
        if scale_method == "StandardScaler":
            scaler       = StandardScaler()
            feat_df[:]   = scaler.fit_transform(feat_df)
            ss.scaler    = scaler

        X_train, X_test, y_train, y_test = train_test_split(
            feat_df, target, test_size=0.2, random_state=42, stratify=target
        )
        ss.X_train, ss.X_test = X_train, X_test
        ss.y_train, ss.y_test = y_train, y_test
        ss.featured_df        = df

        st.success(
            f"Features prepared — Train: {len(X_train):,} rows | Test: {len(X_test):,} rows"
        )

        # Feature distribution viz
        fig = px.box(
            feat_df[selected[:8]], title="Feature Distribution (first 8)",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, width='stretch')

        if st.button("Proceed to Model Training →", type="primary", key="feat_next"):
            _complete_step(4)
            st.rerun()

    elif ss.X_train is not None:
        _ok(f"Features already prepared — {len(ss.X_train):,} train / {len(ss.X_test):,} test rows.")
        if st.button("Proceed to Model Training →", type="primary"):
            _complete_step(4)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def step_train() -> None:
    _step_header(5, "🤖", "Model Training")

    if ss.X_train is None:
        st.error("Complete Feature Engineering first.")
        return

    st.markdown(
        "Three models are trained and compared: "
        "**Logistic Regression** (baseline), "
        "**Random Forest** (ensemble), "
        "**Gradient Boosting** (boosted ensemble)."
    )

    with st.expander("Hyperparameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        lr_c     = col1.slider("LR: regularisation C",      0.01, 10.0, 1.0, 0.01)
        rf_trees = col2.slider("RF: n_estimators",           50, 500, 200, 50)
        gb_lr    = col3.slider("GB: learning_rate",          0.01, 0.5, 0.1, 0.01)
        gb_trees = col3.slider("GB: n_estimators",           50, 300, 100, 50)
        class_wt = col1.checkbox("Use class_weight='balanced'", value=True)

    if st.button("Train All Models", type="primary"):
        cw = "balanced" if class_wt else None

        model_defs = {
            "Logistic Regression": LogisticRegression(
                C=lr_c, max_iter=1_000, class_weight=cw, random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_trees, class_weight=cw, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=gb_trees, learning_rate=gb_lr, random_state=42
            ),
        }

        results = {}
        progress = st.progress(0, "Training models…")
        for i, (name, mdl) in enumerate(model_defs.items()):
            progress.progress((i) / len(model_defs), f"Training {name}…")
            mdl.fit(ss.X_train, ss.y_train)
            y_pred  = mdl.predict(ss.X_test)
            y_proba = mdl.predict_proba(ss.X_test)[:, 1] if hasattr(mdl, "predict_proba") else y_pred

            results[name] = {
                "model":    mdl,
                "y_pred":   y_pred,
                "y_proba":  y_proba,
                "accuracy": accuracy_score(ss.y_test, y_pred),
                "precision":precision_score(ss.y_test, y_pred, zero_division=0),
                "recall":   recall_score(ss.y_test, y_pred, zero_division=0),
                "f1":       f1_score(ss.y_test, y_pred, zero_division=0),
                "roc_auc":  roc_auc_score(ss.y_test, y_proba),
            }

        progress.progress(1.0, "Done!")
        ss.models       = {n: r["model"] for n, r in results.items()}
        ss.model_results= results

        # Feature importance (RF)
        rf  = ss.models.get("Random Forest")
        if rf and hasattr(rf, "feature_importances_"):
            ss.feature_importance = pd.Series(
                rf.feature_importances_, index=ss.feature_cols
            ).sort_values(ascending=False)

        st.success("All 3 models trained successfully!")

    if ss.model_results:
        st.subheader("Training Summary")
        summary = pd.DataFrame([
            {
                "Model":     n,
                "Accuracy":  f"{r['accuracy']:.3f}",
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1":        f"{r['f1']:.3f}",
                "ROC-AUC":   f"{r['roc_auc']:.3f}",
            }
            for n, r in ss.model_results.items()
        ])
        st.dataframe(summary, width='stretch', hide_index=True)

        best = max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
        _ok(f"Best model by ROC-AUC: **{best}** ({ss.model_results[best]['roc_auc']:.3f})")

        if ss.feature_importance is not None:
            st.subheader("Top Feature Importances (Random Forest)")
            top15 = ss.feature_importance.head(15)
            fig = px.bar(
                x=top15.values, y=top15.index, orientation="h",
                color=top15.values, color_continuous_scale="Blues",
                title="Feature Importance", labels={"x": "Importance", "y": "Feature"},
            )
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig, width='stretch')

        if st.button("Proceed to Model Evaluation →", type="primary"):
            _complete_step(5)
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def step_evaluate() -> None:
    _step_header(6, "📊", "Model Evaluation")

    if not ss.model_results:
        st.error("Train models first.")
        return

    chosen = st.selectbox("Select model to evaluate:", list(ss.model_results.keys()))
    r      = ss.model_results[chosen]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    _kpi(c1, f"{r['accuracy']:.3f}",  "Accuracy")
    _kpi(c2, f"{r['precision']:.3f}", "Precision", "green")
    _kpi(c3, f"{r['recall']:.3f}",    "Recall",    "orange")
    _kpi(c4, f"{r['roc_auc']:.3f}",   "ROC-AUC",   "purple")

    col_cm, col_roc = st.columns(2)

    # Confusion matrix
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(ss.y_test, r["y_pred"])
        labels = ["Not Churned", "Churned"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale="Blues", text=cm, texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
            height=320, margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_cm, width='stretch')

    # ROC curves for all models
    with col_roc:
        st.subheader("ROC Curves — All Models")
        fig_roc = go.Figure()
        colours = px.colors.qualitative.Bold
        for i, (name, res) in enumerate(ss.model_results.items()):
            fpr, tpr, _ = roc_curve(ss.y_test, res["y_proba"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res['roc_auc']:.3f})",
                line=dict(color=colours[i], width=2),
            ))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="grey"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=320, margin=dict(t=30, b=10, l=10, r=10), legend=dict(x=0.4, y=0.1),
        )
        st.plotly_chart(fig_roc, width='stretch')

    # Classification report
    st.subheader("Detailed Classification Report")
    report_dict = classification_report(ss.y_test, r["y_pred"], output_dict=True, zero_division=0)
    report_df   = pd.DataFrame(report_dict).T.round(3)
    st.dataframe(report_df, width='stretch')

    # Prediction probability distribution
    st.subheader("Prediction Probability Distribution")
    prob_df = pd.DataFrame({"prob": r["y_proba"], "actual": ss.y_test.values})
    fig_prob = px.histogram(
        prob_df, x="prob", color=prob_df["actual"].astype(str),
        nbins=40, barmode="overlay", opacity=0.7,
        color_discrete_map={"0": "#42a5f5", "1": "#ef5350"},
        labels={"color": "Actual", "prob": "Predicted Probability"},
        title="Churn Probability Distribution",
    )
    fig_prob.update_layout(margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_prob, width='stretch')

    # Insights
    if r["roc_auc"] >= 0.80:
        _ok(f"ROC-AUC {r['roc_auc']:.3f} — strong predictive power.")
    elif r["roc_auc"] >= 0.65:
        _warn(f"ROC-AUC {r['roc_auc']:.3f} — moderate. Consider feature enrichment.")
    else:
        _warn(f"ROC-AUC {r['roc_auc']:.3f} — weak. Review features and data quality.")

    if st.button("Proceed to Bias Detection →", type="primary"):
        _complete_step(6)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BIAS DETECTION (OVERFITTING / UNDERFITTING)
# ─────────────────────────────────────────────────────────────────────────────
def step_bias() -> None:
    _step_header(7, "📉", "Bias Detection — Overfitting & Underfitting")

    if not ss.models:
        st.error("Train models first.")
        return

    st.markdown(
        "**Learning curves** show training score vs. cross-validation score "
        "as the training size grows.  \n"
        "- **Overfit**: high train score, low val score (gap stays large)  \n"
        "- **Underfit**: both scores are low and close together  \n"
        "- **Good fit**: both scores converge at a high value"
    )

    chosen = st.selectbox("Select model:", list(ss.models.keys()))
    mdl    = ss.models[chosen]

    if st.button("Generate Learning Curves", type="primary"):
        X_full = pd.concat([ss.X_train, ss.X_test])
        y_full = pd.concat([ss.y_train, ss.y_test])

        with st.spinner("Computing learning curves (this may take ~30 seconds)…"):
            train_sizes, train_scores, val_scores = learning_curve(
                mdl, X_full, y_full,
                cv=5, scoring="roc_auc",
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1, random_state=42,
            )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig = go.Figure()
        # Train band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill="toself", fillcolor="rgba(66,165,245,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=train_mean,
            mode="lines+markers", name="Train ROC-AUC",
            line=dict(color="#1976d2", width=2),
        ))
        # Val band
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean + val_std, (val_mean - val_std)[::-1]]),
            fill="toself", fillcolor="rgba(239,83,80,0.15)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=val_mean,
            mode="lines+markers", name="CV ROC-AUC",
            line=dict(color="#c62828", width=2),
        ))
        fig.update_layout(
            title=f"Learning Curve — {chosen}",
            xaxis_title="Training Set Size",
            yaxis_title="ROC-AUC Score",
            height=420, margin=dict(t=50, b=30, l=40, r=20),
            legend=dict(x=0.65, y=0.15),
        )
        st.plotly_chart(fig, width='stretch')

        # Diagnosis
        final_gap = float(train_mean[-1] - val_mean[-1])
        final_val = float(val_mean[-1])

        st.subheader("Diagnosis")
        diag_col1, diag_col2 = st.columns(2)
        diag_col1.metric("Final Train AUC", f"{train_mean[-1]:.3f}")
        diag_col2.metric("Final  CV  AUC", f"{val_mean[-1]:.3f}", delta=f"gap {final_gap:.3f}")

        if final_gap > 0.10 and final_val > 0.65:
            verdict = "OVERFITTING"
            _warn(
                f"**Overfitting** detected (train–val gap = {final_gap:.3f}).  \n"
                "Recommendations: reduce model complexity, add regularisation, "
                "collect more data, apply cross-validation."
            )
        elif final_val < 0.60:
            verdict = "UNDERFITTING"
            _warn(
                f"**Underfitting** detected (CV AUC = {final_val:.3f}).  \n"
                "Recommendations: add more features, increase model complexity, "
                "remove excessive regularisation."
            )
        else:
            verdict = "GOOD FIT"
            _ok(
                f"**Good fit** — train–val gap = {final_gap:.3f}, CV AUC = {final_val:.3f}.  \n"
                "The model generalises well to unseen data."
            )

        # Train vs Test comparison bar
        train_auc = ss.model_results[chosen]["roc_auc"]
        train_self_auc = roc_auc_score(
            ss.y_train,
            ss.models[chosen].predict_proba(ss.X_train)[:, 1]
            if hasattr(ss.models[chosen], "predict_proba")
            else ss.models[chosen].predict(ss.X_train),
        )
        fig2 = go.Figure([
            go.Bar(name="Train AUC", x=[chosen], y=[train_self_auc],
                   marker_color="#1976d2"),
            go.Bar(name="Test AUC",  x=[chosen], y=[train_auc],
                   marker_color="#ef5350"),
        ])
        fig2.update_layout(
            barmode="group", title="Train vs Test AUC Comparison",
            height=300, margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, width='stretch')

        # All models comparison
        st.subheader("All Models — Train vs Test AUC")
        rows = []
        for name, mdl_i in ss.models.items():
            tr_auc = roc_auc_score(
                ss.y_train,
                mdl_i.predict_proba(ss.X_train)[:, 1]
                if hasattr(mdl_i, "predict_proba") else mdl_i.predict(ss.X_train),
            )
            te_auc = ss.model_results[name]["roc_auc"]
            rows.append({"Model": name, "Train AUC": round(tr_auc, 3),
                         "Test AUC": round(te_auc, 3), "Gap": round(tr_auc - te_auc, 3),
                         "Verdict": "Overfit" if tr_auc - te_auc > 0.10 else "Good fit"})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if st.button("Proceed to Recommendations →", type="primary"):
        _complete_step(7)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def step_recommendations() -> None:
    _step_header(8, "💡", "Product & Campaign Recommendations")

    if not ss.models:
        st.error("Train models first.")
        return

    df = (ss.featured_df if ss.featured_df is not None else ss.cleaned_df).copy()

    # Use best model for scoring
    best_name  = max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"])
    best_model = ss.models[best_name]

    st.info(f"Churn probability computed using: **{best_name}** (best ROC-AUC)")

    if st.button("Generate Recommendations", type="primary"):
        feat_df = df[ss.feature_cols].copy()
        proba   = best_model.predict_proba(feat_df)[:, 1]
        df["churn_probability"] = proba.round(4)

        # Segment assignment
        def _segment(row) -> str:
            p = row["churn_probability"]
            complaints = row.get("complaint_count", 0) or 0
            data_mb    = row.get("data_volume_mb",   0) or 0
            charges    = row.get("monthly_charges",  0) or 0
            tenure     = row.get("tenure_months",    12) or 12

            if p >= 0.65 and complaints >= 3:
                return "High-Risk / Service Issue"
            if p >= 0.65 and data_mb < 200:
                return "High-Risk / Low Usage"
            if p >= 0.65 and charges > 600:
                return "High-Risk / High Value"
            if p >= 0.40:
                return "At-Risk"
            if tenure < 6:
                return "New Subscriber"
            if data_mb > 3_000 or charges > 500:
                return "Loyal / High Value"
            return "Stable / Standard"

        df["segment"] = df.apply(_segment, axis=1)

        CAMPAIGNS: dict[str, dict] = {
            "High-Risk / Service Issue": {
                "product":  "Priority Support Plan",
                "campaign": "Service Recovery — Personal apology + 3-month fee waiver",
                "urgency":  "CRITICAL",
                "color":    "reco-red",
            },
            "High-Risk / Low Usage": {
                "product":  "Starter Data Bundle 2 GB @ R29",
                "campaign": "Win-Back — Free 1GB data for 2 months",
                "urgency":  "HIGH",
                "color":    "reco-red",
            },
            "High-Risk / High Value": {
                "product":  "Loyalty Rewards Programme",
                "campaign": "Retention — 20% discount + dedicated account manager",
                "urgency":  "HIGH",
                "color":    "reco-amber",
            },
            "At-Risk": {
                "product":  "Flexi Bundle (voice + data + SMS)",
                "campaign": "Proactive Engagement — Usage milestone reward",
                "urgency":  "MEDIUM",
                "color":    "reco-amber",
            },
            "New Subscriber": {
                "product":  "Welcome Pack + 30-day trial upgrade",
                "campaign": "Onboarding Journey — Guided tutorials + first-month bonus",
                "urgency":  "MEDIUM",
                "color":    "reco-blue",
            },
            "Loyal / High Value": {
                "product":  "5G Early Access / Premium Plan Upgrade",
                "campaign": "Upsell — Early 5G access + referral bonus",
                "urgency":  "LOW",
                "color":    "reco-green",
            },
            "Stable / Standard": {
                "product":  "Data Add-On or SMS Bundle",
                "campaign": "Cross-Sell — Personalised data top-up offers",
                "urgency":  "LOW",
                "color":    "reco-green",
            },
        }

        ss.recommendations_df = df

        # Summary counts
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        st.subheader("Customer Segmentation")
        col_pie, col_tbl = st.columns(2)
        with col_pie:
            fig = px.pie(
                seg_counts, names="Segment", values="Count",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Segment Distribution",
            )
            fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, width='stretch')
        with col_tbl:
            st.dataframe(seg_counts, width='stretch', hide_index=True)

        # Segment churn probability
        st.subheader("Average Churn Probability by Segment")
        seg_prob = (
            df.groupby("segment")["churn_probability"]
            .mean().sort_values(ascending=False)
            .reset_index()
        )
        fig2 = px.bar(
            seg_prob, x="churn_probability", y="segment",
            orientation="h", color="churn_probability",
            color_continuous_scale="Reds",
            title="Mean Churn Probability",
            labels={"churn_probability": "Avg P(churn)", "segment": ""},
        )
        fig2.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig2, width='stretch')

        # Campaign cards
        st.subheader("Campaign Recommendations by Segment")
        for seg, camp in CAMPAIGNS.items():
            cnt = int(seg_counts.loc[seg_counts["Segment"] == seg, "Count"].sum())
            if cnt == 0:
                continue
            st.markdown(
                f'<div class="reco-card {camp["color"]}">'
                f"<strong>{seg}</strong> &nbsp;·&nbsp; {cnt:,} subscribers  <br/>"
                f"📦 <em>Product:</em> {camp['product']}  <br/>"
                f"📢 <em>Campaign:</em> {camp['campaign']}  <br/>"
                f"🔴 <em>Urgency:</em> {camp['urgency']}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Export
        csv_buf = io.StringIO()
        df[["churn_probability", "segment"] +
           ([ss.target_col] if ss.target_col in df.columns else [])].to_csv(csv_buf, index=False)
        st.download_button(
            "Download Recommendation CSV",
            data=csv_buf.getvalue(),
            file_name="churn_recommendations.csv",
            mime="text/csv",
        )

    if ss.recommendations_df is not None and st.button("Proceed to Data Story →", type="primary"):
        _complete_step(8)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — DATA STORY / PRESENTATION
# ─────────────────────────────────────────────────────────────────────────────
def step_story() -> None:
    _step_header(9, "🎯", "Data Story — Executive Insights Presentation")

    df       = ss.raw_df
    clean_df = ss.cleaned_df
    reco_df  = ss.recommendations_df
    tgt      = ss.target_col
    best     = max(ss.model_results, key=lambda n: ss.model_results[n]["roc_auc"]) if ss.model_results else None

    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#1565c0,#0d47a1);
    color:#fff;border-radius:16px;padding:28px 32px;margin-bottom:24px;">
  <h2 style="margin:0 0 6px 0;">📡 Telecom Subscriber Analytics</h2>
  <p style="margin:0;opacity:.85;font-size:1rem;">
    Data Story — {ss.dataset_name} &nbsp;·&nbsp; {datetime.now().strftime("%B %Y")}
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Chapter 1: Data Quality ───────────────────────────────────────────────
    st.subheader("1. Data Quality Summary")
    if df is not None:
        c1, c2, c3, c4, c5 = st.columns(5)
        _kpi(c1, f"{df.shape[0]:,}",           "Raw Records")
        _kpi(c2, f"{df.shape[1]}",             "Features",     "green")
        _kpi(c3, f"{df.isnull().mean().mean()*100:.1f}%", "Null Rate", "orange")
        _kpi(c4, f"{df.duplicated().sum():,}",  "Duplicates",  "orange")
        rows_cleaned = len(clean_df) if clean_df is not None else len(df)
        _kpi(c5, f"{rows_cleaned:,}",           "Clean Records","green")

        for insight in ss.eda_insights[:3]:
            _insight(insight)
        for entry in ss.clean_log[-3:]:
            _ok(entry)

    # ── Chapter 2: Churn Profile ──────────────────────────────────────────────
    st.subheader("2. Churn Profile")
    if tgt and df is not None and tgt in df.columns:
        if pd.api.types.is_numeric_dtype(df[tgt]):
            churn_rate = float(df[tgt].mean() * 100)
            non_churn  = 100 - churn_rate

            col_donut, col_stats = st.columns([1, 2])
            with col_donut:
                fig = go.Figure(go.Pie(
                    labels=["Churned", "Retained"],
                    values=[churn_rate, non_churn],
                    hole=0.55,
                    marker_colors=["#ef5350", "#42a5f5"],
                ))
                fig.update_layout(
                    height=280, margin=dict(t=10, b=10, l=10, r=10),
                    annotations=[dict(text=f"{churn_rate:.1f}%\nChurn", showarrow=False,
                                      font=dict(size=16, color="#ef5350"))],
                )
                st.plotly_chart(fig, width='stretch')

            with col_stats:
                total    = len(df)
                churned  = int(df[tgt].sum())
                retained = total - churned
                if churn_rate > 25:
                    _warn(f"Churn rate of **{churn_rate:.1f}%** is above the 25% critical threshold.")
                elif churn_rate > 15:
                    _warn(f"Churn rate of **{churn_rate:.1f}%** is elevated. Proactive retention recommended.")
                else:
                    _ok(f"Churn rate of **{churn_rate:.1f}%** is within acceptable range.")

                st.metric("Total Subscribers", f"{total:,}")
                st.metric("Churned",  f"{churned:,}",  delta=f"-{churn_rate:.1f}%", delta_color="inverse")
                st.metric("Retained", f"{retained:,}", delta=f"+{non_churn:.1f}%")

    # ── Chapter 3: Key Drivers ────────────────────────────────────────────────
    st.subheader("3. Churn Drivers")
    if ss.feature_importance is not None:
        top8 = ss.feature_importance.head(8)
        fig  = px.bar(
            x=top8.values, y=top8.index, orientation="h",
            color=top8.values, color_continuous_scale="RdYlGn_r",
            title="Top 8 Churn Drivers (Random Forest)",
            labels={"x": "Importance Score", "y": "Feature"},
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig, width='stretch')
        _insight(f"Most impactful churn driver: **{top8.index[0]}**")
    else:
        _warn("Feature importance not available — complete model training.")

    # ── Chapter 4: Model Performance ─────────────────────────────────────────
    st.subheader("4. Model Performance Scorecard")
    if ss.model_results:
        perf_rows = []
        for n, r in ss.model_results.items():
            verdict = ""
            train_auc = roc_auc_score(
                ss.y_train,
                ss.models[n].predict_proba(ss.X_train)[:, 1]
                if hasattr(ss.models[n], "predict_proba") else ss.models[n].predict(ss.X_train),
            )
            gap = train_auc - r["roc_auc"]
            if gap > 0.10:
                verdict = "Overfitting"
            elif r["roc_auc"] < 0.60:
                verdict = "Underfitting"
            else:
                verdict = "Good Fit"
            perf_rows.append({
                "Model":      n,
                "Accuracy":   f"{r['accuracy']:.3f}",
                "F1 Score":   f"{r['f1']:.3f}",
                "ROC-AUC":    f"{r['roc_auc']:.3f}",
                "Train AUC":  f"{train_auc:.3f}",
                "Bias":       verdict,
                "Recommended": "Yes" if n == best else "",
            })

        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(
            perf_df.style.map(
                lambda v: "background-color:#c8e6c9" if v == "Yes" else "",
                subset=["Recommended"],
            ).map(
                lambda v: "color:#ef5350" if v == "Overfitting" else
                           "color:#ff9800" if v == "Underfitting" else
                           "color:#4caf50",
                subset=["Bias"],
            ),
            width='stretch', hide_index=True,
        )
        _ok(f"Recommended model: **{best}** (ROC-AUC = {ss.model_results[best]['roc_auc']:.3f})")

    # ── Chapter 5: Recommendations Summary ───────────────────────────────────
    if reco_df is not None:
        st.subheader("5. Campaign Recommendations Summary")
        seg_summary = (
            reco_df.groupby("segment")["churn_probability"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "Subscribers", "mean": "Avg Churn P"})
            .sort_values("Avg Churn P", ascending=False)
        )
        seg_summary["Avg Churn P"] = seg_summary["Avg Churn P"].round(3)

        fig_seg = px.scatter(
            seg_summary,
            x="Subscribers", y="Avg Churn P",
            size="Subscribers", color="segment",
            text="segment", title="Segment Map",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_seg.update_traces(textposition="top center")
        fig_seg.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_seg, width='stretch')

        at_risk = int(reco_df[reco_df["churn_probability"] >= 0.5].shape[0])
        _warn(
            f"**{at_risk:,} subscribers** have a churn probability ≥ 50% and should be "
            "prioritised for retention campaigns immediately."
        )

    # ── Chapter 6: Action Plan ────────────────────────────────────────────────
    st.subheader("6. Recommended Action Plan")
    actions = [
        ("🚨 Immediate (0-7 days)",   "#ffcdd2",
         "Contact all 'High-Risk / Service Issue' subscribers. "
         "Escalate complaints. Deploy service recovery scripts."),
        ("⚡ Short-term (1-4 weeks)",  "#fff9c4",
         "Launch Win-Back data bundle campaign for 'High-Risk / Low Usage'. "
         "Set up automated churn-score alerts in CRM."),
        ("📈 Medium-term (1-3 months)", "#bbdefb",
         "Deploy Flexi Bundle cross-sell for 'At-Risk' segment. "
         "Onboarding journey automation for 'New Subscribers'."),
        ("🌱 Long-term (3-12 months)", "#c8e6c9",
         "Premium 5G upgrade campaign for 'Loyal / High Value'. "
         "Integrate ML model scores into real-time decisioning engine."),
    ]
    for title, bg, detail in actions:
        st.markdown(
            f'<div style="background:{bg};border-radius:8px;padding:14px;margin:8px 0;">'
            f"<strong>{title}</strong><br/>{detail}</div>",
            unsafe_allow_html=True,
        )

    # ── Export presentation as JSON ───────────────────────────────────────────
    st.subheader("Export")
    report = {
        "generated_at":  datetime.now().isoformat(),
        "dataset":       ss.dataset_name,
        "records":       int(df.shape[0]) if df is not None else 0,
        "churn_rate":    float(df[tgt].mean() * 100) if (tgt and df is not None and tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt])) else None,
        "best_model":    best,
        "best_roc_auc":  float(ss.model_results[best]["roc_auc"]) if best else None,
        "eda_insights":  ss.eda_insights,
        "clean_log":     ss.clean_log,
        "top_features":  ss.feature_importance.head(5).to_dict() if ss.feature_importance is not None else {},
    }
    st.download_button(
        "Download Insight Report (JSON)",
        data=json.dumps(report, indent=2, default=str),
        file_name=f"telco_insights_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

    st.balloons()
    _ok("Pipeline complete! All 9 steps finished. Your data story is ready.")
    _complete_step(9)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
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

STEP_FNS[ss.current_step]()

