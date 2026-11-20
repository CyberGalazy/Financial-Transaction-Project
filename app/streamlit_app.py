"""Fraud Detection dashboard.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import config, explain, features  # noqa: E402
from fraud.scoring import REQUIRED_COLUMNS, Artifacts, score_transactions  # noqa: E402

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}
MODEL_ORDER = ["Rule-based baseline", "Isolation Forest", "Logistic Regression", "LightGBM"]


# ------------------------------------------------------------------ loaders
@st.cache_resource(show_spinner="Loading models ...")
def load_artifacts() -> Artifacts:
    return Artifacts.load()


@st.cache_resource(show_spinner="Building explainer ...")
def load_explainer():
    return explain.explainer(load_artifacts().lgbm)


@st.cache_data(show_spinner="Loading scored transactions ...")
def load_scored() -> pd.DataFrame:
    return pd.read_parquet(config.SCORED_TEST_PARQUET)


@st.cache_data
def load_json(name: str):
    return json.loads((config.MODELS_DIR / name).read_text())


@st.cache_data
def load_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(config.MODELS_DIR / name, **kw)


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


if not (config.MODELS_DIR / "lgbm.joblib").exists():
    st.error("No trained model found. Run `python scripts/train.py` first.")
    st.stop()

metrics = load_json("metrics.json")
thresholds = load_json("thresholds.json")["LightGBM"]

st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio("Page", ["Overview", "Flagged transactions", "Investigate a transaction", "Score new data"])
st.sidebar.caption(
    "LightGBM on transaction, behavioural, card and customer features, with an Isolation Forest "
    "anomaly score as an input. Evaluated on 2019 transactions the models never saw.")
st.sidebar.markdown(f"**Flag threshold:** {thresholds['flag']:.3f}  \n**Review threshold:** {thresholds['review']:.3f}")


# ------------------------------------------------------------------ overview
if page == "Overview":
    st.title("Fraud detection — model overview")
    st.markdown(
        "Digital transactions are growing and fraud causes significant losses. Static rules miss "
        "complex patterns, so this system learns from transaction behaviour, scores anomalies and "
        "flags suspicious activity. Results below are on the **2019 hold-out year**.")

    m = metrics["LightGBM"]
    c = st.columns(5)
    c[0].metric("Fraud caught (recall)", fmt_pct(m["recall"]))
    c[1].metric("Precision of flags", fmt_pct(m["precision"]))
    c[2].metric("PR-AUC", f"{m['pr_auc']:.3f}", f"{m['pr_auc'] - metrics['Rule-based baseline']['pr_auc']:+.3f} vs rules")
    c[3].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c[4].metric("Transactions flagged", fmt_pct(m["flag_rate"]))

    st.subheader("Model comparison")
    table = pd.DataFrame(metrics).T.loc[MODEL_ORDER]
    show = table[["pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "f1", "flag_rate", "tp", "fp", "fn"]]
    show.columns = ["PR-AUC", "ROC-AUC", "Recall @1% FPR", "Precision", "Recall", "F1", "Flag rate",
                    "Frauds caught", "False alarms", "Frauds missed"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[:7]} | {c: "{:,.0f}" for c in show.columns[7:]})
                 .highlight_max(subset=["PR-AUC", "ROC-AUC", "F1"], color="#c7e9c0"), width="stretch")
    st.caption("PR-AUC is the headline metric: fraud is ~0.15% of transactions, so ROC-AUC alone is misleading.")

    curves = load_json("curves.json")
    left, right = st.columns(2)
    fig = go.Figure()
    for name in MODEL_ORDER:
        fig.add_scatter(x=curves[name]["pr"]["recall"], y=curves[name]["pr"]["precision"], name=name, mode="lines")
    fig.update_layout(title="Precision–Recall (test)", xaxis_title="Recall", yaxis_title="Precision", height=420)
    left.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for name in MODEL_ORDER:
        fig.add_scatter(x=curves[name]["roc"]["fpr"], y=curves[name]["roc"]["tpr"], name=name, mode="lines")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), showlegend=False)
    fig.update_layout(title="ROC (test)", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    monthly = load_csv("fraud_by_month.csv", parse_dates=["date"])
    fig = px.line(monthly, x="date", y="fraud_rate", title="Fraud rate by month (labelled transactions)")
    fig.add_vline(x=pd.Timestamp(config.TRAIN_END), line_dash="dash", line_color="grey")
    fig.add_vline(x=pd.Timestamp(config.VALID_END), line_dash="dash", line_color="black")
    fig.update_yaxes(tickformat=".2%")
    left.plotly_chart(fig, width="stretch")
    left.caption("Dashed lines: end of training data, end of validation. Fraud patterns shift over time, "
                 "so models must be retrained on recent data.")
    imp = load_csv("feature_importance.csv").head(15)
    imp["label"] = imp["feature"].map(lambda f: explain.FEATURE_LABELS.get(f, f))
    fig = px.bar(imp[::-1], x="gain", y="label", orientation="h", title="Top features (LightGBM gain)")
    fig.update_layout(yaxis_title="", height=450)
    right.plotly_chart(fig, width="stretch")

    st.subheader("Why rules are not enough")
    rules = load_csv("rule_stats.csv", index_col=0)
    rules.columns = ["Fires on % of txns", "Precision", "Recall"]
    st.dataframe(rules.style.format("{:.2%}"), width="stretch")
    st.caption("Each static rule either fires on many legitimate transactions or misses most fraud; "
               "the ML model combines dozens of signals into one calibrated risk score.")


# ------------------------------------------------------------------ flagged list
elif page == "Flagged transactions":
    st.title("Flagged transactions (2019 hold-out)")
    df = load_scored()
    f1, f2, f3, f4 = st.columns(4)
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium"])
    dmin, dmax = df["date"].min().date(), df["date"].max().date()
    dates = f2.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    min_amt = f3.number_input("Min amount ($)", value=0.0, step=50.0)
    cats = f4.multiselect("Merchant category", sorted(df["mcc_description"].astype(str).unique()))

    q = df[df["risk_level"].isin(levels) & (df["amount"] >= min_amt)]
    if isinstance(dates, tuple) and len(dates) == 2:
        q = q[(q["date"].dt.date >= dates[0]) & (q["date"].dt.date <= dates[1])]
    if cats:
        q = q[q["mcc_description"].astype(str).isin(cats)]
    q = q.sort_values("fraud_score", ascending=False)

    c = st.columns(4)
    c[0].metric("Transactions shown", f"{len(q):,}")
    c[1].metric("Confirmed fraud among them", f"{int(q['is_fraud'].sum()):,}")
    c[2].metric("Precision", fmt_pct(q["is_fraud"].mean()) if len(q) else "–")
    c[3].metric("Amount at risk", f"${q.loc[q['risk_level'] == 'High', 'amount'].sum():,.0f}")

    cols = ["id", "date", "card_id", "client_id", "amount", "merchant_city", "merchant_state",
            "mcc_description", "use_chip", "errors", "fraud_score", "risk_level", "is_fraud"]
    view = q[cols].head(2000).rename(columns={"is_fraud": "actual_fraud", "mcc_description": "category"})
    st.dataframe(
        view.style.format({"fraud_score": "{:.3f}", "amount": "${:,.2f}"})
        .map(lambda v: f"color: {RISK_COLORS.get(v, '')}; font-weight: 600", subset=["risk_level"]),
        width="stretch", height=520)
    st.caption("Showing the top 2,000 by score. `actual_fraud` is the ground-truth label, shown for evaluation.")
    st.download_button("Download filtered list (CSV)", q[cols].to_csv(index=False), "flagged_transactions.csv")


# ------------------------------------------------------------------ investigate one
elif page == "Investigate a transaction":
    st.title("Investigate a transaction")
    df = load_scored()
    top = df.nlargest(50, "fraud_score")["id"].tolist()
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Pick one of the 50 highest-risk transactions", top)
    typed = c2.text_input("…or enter any 2019 transaction id", "")
    txn_id = int(typed) if typed.strip().isdigit() else pick
    row = df[df["id"] == txn_id]
    if row.empty:
        st.warning("Transaction id not found in the 2019 hold-out set.")
        st.stop()
    r = row.iloc[0]

    c = st.columns(4)
    c[0].metric("Fraud score", f"{r['fraud_score']:.3f}")
    c[1].markdown(f"### <span style='color:{RISK_COLORS[r['risk_level']]}'>{r['risk_level']} risk</span>",
                  unsafe_allow_html=True)
    c[2].metric("Amount", f"${r['amount']:,.2f}")
    c[3].metric("Actual label", "Fraud" if r["is_fraud"] == 1 else "Legitimate")

    st.markdown(
        f"**{r['date']:%Y-%m-%d %H:%M}** · card `{r['card_id']}` · client `{r['client_id']}` · "
        f"{r['use_chip']} · {r['merchant_city']}, {r['merchant_state'] if pd.notna(r['merchant_state']) else '—'} · "
        f"{r['mcc_description']} · errors: {r['errors'] or 'none'}")

    X = features.model_matrix(row)[config.MODEL_FEATURES]
    reasons = explain.top_reasons(load_explainer(), X, k=10)
    reasons["direction"] = np.where(reasons["contribution"] > 0, "raises risk", "lowers risk")
    reasons["text"] = reasons["label"] + " = " + reasons["value"].map(lambda v: f"{v:,.2f}".rstrip("0").rstrip("."))
    fig = px.bar(reasons[::-1], x="contribution", y="text", color="direction", orientation="h",
                 color_discrete_map={"raises risk": "#d62728", "lowers risk": "#1f77b4"},
                 title="Why this score? (SHAP contribution to log-odds)")
    fig.update_layout(yaxis_title="", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Card activity around this transaction")
    hist = df[(df["card_id"] == r["card_id"]) &
              (df["date"].between(r["date"] - pd.Timedelta(days=7), r["date"] + pd.Timedelta(days=2)))]
    st.dataframe(hist[["id", "date", "amount", "merchant_city", "merchant_state", "mcc_description",
                       "fraud_score", "risk_level", "is_fraud"]].sort_values("date")
                 .style.format({"fraud_score": "{:.3f}", "amount": "${:,.2f}"}), width="stretch")


# ------------------------------------------------------------------ score new data
else:
    st.title("Score new transactions")
    st.markdown(
        "Upload a CSV with the same columns as `transactions_data.csv` "
        f"(`{', '.join(REQUIRED_COLUMNS)}`). Each card's earlier history is pulled from the dataset so "
        "behavioural features are computed exactly as in training.")
    up = st.file_uploader("Transactions CSV", type="csv")
    use_example = st.button("…or score 200 random 2019 transactions as an example")

    raw = None
    if up is not None:
        raw = pd.read_csv(up)
    elif use_example:
        sample_ids = load_scored()["id"].sample(200, random_state=np.random.randint(1e6))
        raw = pd.read_parquet(config.CLEAN_TXN_PARQUET, filters=[("id", "in", sample_ids.tolist())])

    if raw is not None:
        try:
            with st.spinner(f"Scoring {len(raw):,} transactions ..."):
                out = score_transactions(raw, load_artifacts())
        except ValueError as e:
            st.error(str(e))
            st.stop()
        c = st.columns(3)
        c[0].metric("Scored", f"{len(out):,}")
        c[1].metric("Flagged (high risk)", f"{int(out['flagged'].sum()):,}")
        c[2].metric("Needs review (medium)", f"{int((out['risk_level'] == 'Medium').sum()):,}")
        cols = ["id", "date", "card_id", "amount", "merchant_city", "merchant_state", "mcc_description",
                "fraud_score", "risk_level", "top_reasons"]
        res = out[cols].sort_values("fraud_score", ascending=False)
        st.dataframe(res.style.format({"fraud_score": "{:.3f}", "amount": "${:,.2f}"})
                     .map(lambda v: f"color: {RISK_COLORS.get(v, '')}; font-weight: 600", subset=["risk_level"]),
                     width="stretch", height=500)
        st.download_button("Download scores (CSV)", res.to_csv(index=False), "fraud_scores.csv")
