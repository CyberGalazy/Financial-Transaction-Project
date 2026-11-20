"""Build the project report PDF from the trained artifacts.

Usage:  python scripts/build_report.py   ->  docs/Fraud_Detection_Project_Report.pdf
Run scripts/train.py first; every number and chart is read from models/ and data/processed/.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraud import config, explain, features, models  # noqa: E402

OUT_DIR = ROOT / "docs"
FIG_DIR = OUT_DIR / "figures"
OUT_PDF = OUT_DIR / "Fraud_Detection_Project_Report.pdf"

# ------------------------------------------------------------------ look & feel
INK = "#1f2937"
MUTED = "#6b7280"
ACCENT = "#1d4ed8"
FRAUD = "#dc2626"
LEGIT = "#2563eb"
MODEL_COLORS = {"Rule-based baseline": "#9ca3af", "Isolation Forest": "#f59e0b",
                "Logistic Regression": "#10b981", "LightGBM": "#1d4ed8"}
MODEL_ORDER = list(MODEL_COLORS)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.edgecolor": "#d1d5db",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#eef0f3",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 200,
})


def _register_fonts() -> tuple[str, str]:
    base = Path("/System/Library/Fonts/Supplemental")
    try:
        pdfmetrics.registerFont(TTFont("Arial", str(base / "Arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(base / "Arial Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Italic", str(base / "Arial Italic.ttf")))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")
        return "Arial", "Arial-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


FONT, FONT_B = _register_fonts()
ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName=FONT_B, fontSize=26, leading=32, textColor=INK,
                            alignment=TA_CENTER),
    "subtitle": ParagraphStyle("st", fontName=FONT, fontSize=13, leading=18, textColor=MUTED, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=16, leading=20, textColor=ACCENT, spaceBefore=10,
                         spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=11.5, leading=15, textColor=INK, spaceBefore=8,
                         spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("b", fontName=FONT, fontSize=9.7, leading=14, textColor=INK, spaceAfter=6),
    "bullet": ParagraphStyle("bl", fontName=FONT, fontSize=9.7, leading=13.5, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=2),
    "caption": ParagraphStyle("c", fontName=FONT, fontSize=8, leading=10.5, textColor=MUTED, alignment=TA_CENTER,
                              spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.3, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=FONT_B, fontSize=7.8, leading=10.5, textColor="white"),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.3, leading=11, textColor=INK,
                           backColor="#f3f4f6", borderPadding=6, leftIndent=6, rightIndent=6, spaceAfter=10),
    "callout": ParagraphStyle("co", fontName=FONT, fontSize=9.7, leading=14, textColor=INK, backColor="#eff6ff",
                              borderColor="#bfdbfe", borderWidth=0.8, borderPadding=8, leftIndent=8, rightIndent=8,
                              spaceBefore=14, spaceAfter=16),
}
CONTENT_W = A4[0] - 4 * cm


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, col_widths=None, header=True, align_right_from=None, zebra=True, highlight_row=None):
    data = [[Paragraph(str(c), S["cellb"] if header and i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#d1d5db")),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f8fafc")))
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#dbeafe")))
    t.setStyle(TableStyle(style))
    return t


def figure(path, width_frac=1.0, caption=None):
    img = Image(str(path))
    w = CONTENT_W * width_frac
    img.drawHeight = img.drawHeight * w / img.drawWidth
    img.drawWidth = w
    parts = [img]
    if caption:
        parts.append(P(caption, "caption"))
    return KeepTogether(parts)


def save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ data
def load():
    md = config.MODELS_DIR
    d = {
        "metrics": json.loads((md / "metrics.json").read_text()),
        "curves": json.loads((md / "curves.json").read_text()),
        "thresholds": json.loads((md / "thresholds.json").read_text()),
        "run": json.loads((md / "run_info.json").read_text()),
        "imp": pd.read_csv(md / "feature_importance.csv"),
        "rules": pd.read_csv(md / "rule_stats.csv", index_col=0),
        "monthly": pd.read_csv(md / "fraud_by_month.csv", parse_dates=["date"]),
        "scored": pd.read_parquet(config.SCORED_TEST_PARQUET),
        "feat": pd.read_parquet(config.FEATURES_PARQUET, columns=[
            "date", "amount", "hour", "is_online", "is_foreign", "mcc_description", "is_fraud"]),
    }
    f = d["feat"]
    f["channel"] = np.select([f.is_online == 1, f.is_foreign == 1], ["Online", "Foreign merchant"], "In-person (US)")
    d["n_txn_total"] = sum(1 for _ in open(config.TRANSACTIONS_CSV)) - 1
    d["n_users"] = len(pd.read_csv(config.USERS_CSV, usecols=[0]))
    d["n_cards"] = len(pd.read_csv(config.CARDS_CSV, usecols=[0]))
    return d


# ------------------------------------------------------------------ charts
def chart_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    steps = [
        ("Raw data", "users, cards,\n13.3M transactions,\nfraud labels"),
        ("Clean & merge", "strip $, parse dates,\njoin card + user"),
        ("Features", "transaction, past-only\nbehaviour, card,\ncustomer"),
        ("Anomaly score", "Isolation Forest\n(unsupervised)"),
        ("Classifier", "LightGBM\n(+ baselines)"),
        ("Decide & explain", "risk tiers from\nvalidation; SHAP\nreasons"),
        ("Serve", "Streamlit app:\nflag, investigate,\nscore CSVs"),
    ]
    w, gap = 1.22, 0.2
    for i, (title, body) in enumerate(steps):
        x = 0.05 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 1.9, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eff6ff" if i not in (3, 4) else "#dbeafe", ec=ACCENT, lw=1))
        ax.text(x + w / 2, 1.95, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, 1.1, body, ha="center", va="center", fontsize=7, color=MUTED, linespacing=1.35)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.02, 1.3), xytext=(x + w + 0.02, 1.3),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2))
    return save(fig, "pipeline")


def chart_class_balance_and_amount(d):
    f = d["feat"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    smp = pd.concat([f[f.is_fraud == 1], f[f.is_fraud == 0].sample(300_000, random_state=0)])
    bins = np.linspace(-200, 1500, 70)
    for lab, col, name in [(0, LEGIT, "Legitimate"), (1, FRAUD, "Fraud")]:
        axes[0].hist(smp.loc[smp.is_fraud == lab, "amount"].clip(-200, 1500), bins=bins, density=True,
                     alpha=0.55, color=col, label=name)
    axes[0].set(title="Transaction amount: fraud vs legitimate", xlabel="Amount ($, clipped to 1,500)",
                ylabel="Density")
    axes[0].legend(frameon=False)
    by_hour = f.groupby("hour").is_fraud.mean() * 100
    axes[1].bar(by_hour.index, by_hour.values, color=ACCENT)
    axes[1].set(title="Fraud rate by hour of day", xlabel="Hour", ylabel="% fraudulent")
    fig.tight_layout()
    return save(fig, "amount_hour")


def chart_drift(d):
    f = d["feat"]
    drift = f.assign(year=f.date.dt.year).pivot_table(index="year", columns="channel", values="is_fraud",
                                                      aggfunc="mean") * 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    cols = {"In-person (US)": "#6b7280", "Online": "#10b981", "Foreign merchant": FRAUD}
    for c, col in cols.items():
        axes[0].plot(drift.index, drift[c], marker="o", ms=4, color=col, label=c)
    axes[0].set(title="Fraud rate by channel and year", ylabel="% fraudulent", xlabel="Year")
    axes[0].legend(frameon=False)
    m = d["monthly"]
    axes[1].plot(m["date"], m["fraud_rate"] * 100, color=ACCENT, lw=1.2)
    axes[1].axvspan(pd.Timestamp(config.TRAIN_END), pd.Timestamp(config.VALID_END), color="#f59e0b", alpha=0.18,
                    lw=0, label="validation")
    axes[1].axvspan(pd.Timestamp(config.VALID_END), m["date"].max() + pd.offsets.MonthEnd(1), color=FRAUD,
                    alpha=0.15, lw=0, label="test")
    axes[1].legend(frameon=False, loc="upper left", fontsize=7.5, title="unshaded = training", title_fontsize=7)
    axes[1].set(title="Monthly fraud rate and the time-based split", ylabel="% fraudulent")
    fig.tight_layout()
    return save(fig, "drift"), drift


def chart_curves(d):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name in MODEL_ORDER:
        c, m = d["curves"][name], d["metrics"][name]
        lw = 2.2 if name == "LightGBM" else 1.3
        axes[0].plot(c["pr"]["recall"], c["pr"]["precision"], color=MODEL_COLORS[name], lw=lw,
                     label=f"{name} (AP {m['pr_auc']:.3f})")
        axes[1].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=MODEL_COLORS[name], lw=lw,
                     label=f"{name} (AUC {m['roc_auc']:.3f})")
    axes[0].set(title="Precision-Recall curve (2019 test)", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    axes[1].plot([0, 1], [0, 1], ls=":", color=MUTED)
    axes[1].set(title="ROC curve (2019 test)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set_xscale("log"); axes[1].set_xlim(1e-5, 1)
    for a in axes:
        a.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", framealpha=1, fontsize=7.2,
                 loc="lower left" if a is axes[0] else "upper left")
    fig.tight_layout()
    return save(fig, "curves")


def chart_confusion_and_tiers(d):
    s = d["scored"]
    thr = d["thresholds"]["LightGBM"]["flag"]
    pred = s.fraud_score >= thr
    cm_ = np.array([[((~pred) & (s.is_fraud == 0)).sum(), (pred & (s.is_fraud == 0)).sum()],
                    [((~pred) & (s.is_fraud == 1)).sum(), (pred & (s.is_fraud == 1)).sum()]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    ax.imshow(np.log10(cm_ + 1), cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_[i, j]:,}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if cm_[i, j] > 10_000 else INK)
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"]); ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_title("LightGBM confusion matrix (High-risk threshold)")
    tiers = s.groupby("risk_level").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum")).reindex(
        ["High", "Medium", "Low"])
    tiers["legit"] = tiers.n - tiers.fraud
    ax = axes[1]
    y = np.arange(3)
    ax.barh(y, tiers.fraud, color=FRAUD, label="Fraud")
    ax.barh(y, tiers.legit.clip(upper=2500), left=tiers.fraud, color="#cbd5e1", label="Legitimate")
    for i, (tier, r) in enumerate(tiers.iterrows()):
        ax.text(min(r.n, 2500 + r.fraud) + 40, i, f"{int(r.fraud):,} fraud / {int(r.n):,} txns ({r.fraud / r.n:.1%})",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, ["High", "Medium", "Low"]); ax.invert_yaxis()
    ax.set_xlim(0, 4300)
    ax.set_title("Risk tiers on 2019 (Low bar truncated)"); ax.set_xlabel("Transactions")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return save(fig, "confusion_tiers"), cm_, tiers


def chart_importance(d):
    imp = d["imp"].head(15).copy()
    imp["label"] = imp.feature.map(lambda x: explain.FEATURE_LABELS.get(x, x))
    imp["share"] = imp.gain / d["imp"].gain.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(imp.label[::-1], imp.share[::-1], color=ACCENT)
    ax.set(title="Top 15 features (share of LightGBM total gain)", xlabel="% of total gain")
    fig.tight_layout()
    return save(fig, "importance")


def chart_example(d):
    import joblib
    s = d["scored"]
    case = s[(s.is_fraud == 1) & (s.risk_level == "High")].nlargest(1, "fraud_score")
    lgbm = joblib.load(config.MODELS_DIR / "lgbm.joblib")
    reasons = explain.top_reasons(explain.explainer(lgbm), features.model_matrix(case)[config.MODEL_FEATURES], k=8)

    def fmt(v):
        return str(int(v)) if float(v).is_integer() else f"{v:,.2f}".rstrip("0").rstrip(".")
    reasons["text"] = reasons.label + " = " + reasons.value.map(fmt)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    r = reasons[::-1]
    ax.barh(r.text, r.contribution, color=[FRAUD if v > 0 else LEGIT for v in r.contribution])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(title="Why was this transaction flagged? (SHAP contribution to log-odds)",
           xlabel="red = raises risk, blue = lowers risk")
    fig.tight_layout()
    return save(fig, "example_shap"), case.iloc[0]


# ------------------------------------------------------------------ document
def build():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    m, run, thr = d["metrics"], d["run"], d["thresholds"]["LightGBM"]
    lg, rb, lr, ifo = m["LightGBM"], m["Rule-based baseline"], m["Logistic Regression"], m["Isolation Forest"]
    split = run["split_sizes"]
    sc = d["scored"]
    n_test_fraud = int(sc.is_fraud.sum())

    fig_pipe = chart_pipeline()
    fig_amt = chart_class_balance_and_amount(d)
    fig_drift, drift = chart_drift(d)
    fig_curves = chart_curves(d)
    fig_cm, cm_, tiers = chart_confusion_and_tiers(d)
    fig_imp = chart_importance(d)
    fig_ex, case = chart_example(d)
    hm = tiers.loc[["High", "Medium"]]
    review_recall = hm.fraud.sum() / n_test_fraud

    st = []
    # ---------------- title page
    st += [Spacer(1, 5.5 * cm), P("Fraud Detection in Digital<br/>Financial Transactions", "title"), Spacer(1, 10),
           P("A machine-learning system that analyses transaction behaviour,<br/>detects anomalies and flags "
             "potentially fraudulent activity", "subtitle"), Spacer(1, 2.2 * cm)]
    kpis = [[f"{lg['pr_auc']:.2f}", f"{lg['recall_at_1pct_fpr']:.0%}", f"{lg['precision']:.0%}",
             f"{rb['fp'] / max(lg['fp'], 1):.0f}x"],
            ["PR-AUC (rules: %.2f)" % rb["pr_auc"], "of fraud caught at a 1%<br/>false-positive rate",
             "precision of<br/>high-risk flags", "fewer false alarms<br/>than the rule system"]]
    kt = Table([[Paragraph(f"<font size=22 color='{ACCENT}'><b>{v}</b></font>", S["subtitle"]) for v in kpis[0]],
                [Paragraph(f"<font size=8.5>{v}</font>", ParagraphStyle("k", parent=S["subtitle"], leading=11)) for v in kpis[1]]],
               colWidths=[CONTENT_W / 4] * 4)
    kt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 1), (-1, 1), 6)]))
    st += [kt, Spacer(1, 3.2 * cm),
           P(f"Project report &nbsp;&middot;&nbsp; {date.today():%d %B %Y}", "subtitle"),
           P("Evaluated on 2019 transactions never used for training or tuning", "subtitle"), PageBreak()]

    # ---------------- 1. executive summary
    st += [P("1. Executive summary", "h1"),
           P("Digital payments keep growing, and so do the losses from fraud. Most banks still rely on hand-written "
             "rules such as <i>block very large amounts</i> or <i>flag foreign transactions</i>. Rules are easy to "
             "understand, but they cannot weigh many weak signals together, they raise large numbers of false "
             "alarms, and they stop working when fraudsters change tactics."),
           P(f"This project builds a machine-learning fraud detection system on {d['n_txn_total'] / 1e6:.1f} million "
             f"card transactions from {d['n_users']:,} customers and {d['n_cards']:,} cards (2010-2019). For every "
             "transaction it computes how the card normally behaves, scores how unusual the new transaction is, and "
             "outputs a fraud probability with a risk tier (High / Medium / Low) and a plain-language explanation."),
           P("Headline results on the 2019 hold-out year", "h2")]
    st += bullets([
        f"The main model (LightGBM) reaches a <b>PR-AUC of {lg['pr_auc']:.3f}</b>, against {rb['pr_auc']:.3f} for "
        f"the rule-based system, {lr['pr_auc']:.3f} for logistic regression and {ifo['pr_auc']:.3f} for pure anomaly "
        "detection.",
        f"At a 1% false-positive rate it catches <b>{lg['recall_at_1pct_fpr']:.0%} of fraud</b>.",
        f"At the High-risk threshold, <b>{lg['precision']:.0%} of flags are real fraud</b>: {lg['tp']:,} frauds caught "
        f"with {lg['fp']:,} false alarms. The rule system needs {rb['fp']:,} false alarms to catch its frauds.",
        f"Adding the Medium (review) tier catches <b>{review_recall:.0%} of all fraud</b> while sending only "
        f"{int(hm.n.sum()):,} of {len(sc):,} transactions ({hm.n.sum() / len(sc):.2%}) for review.",
        "Every flag comes with the top reasons behind it (SHAP), and a Streamlit dashboard lets analysts browse "
        "alerts, investigate single transactions and score new files.",
    ])
    st += [P("Two findings shaped the design: fraud patterns in this data <b>shift strongly over time</b>, and "
             "standard tricks for imbalanced data (up-weighting the rare class) <b>hurt</b> here. Both are covered in "
             "section 9.", "callout")]

    # ---------------- 2. problem
    st += [P("2. Problem statement", "h1"),
           P("<i>Digital financial transactions are increasing, and fraudulent activities can cause significant "
             "financial losses. Traditional rule-based detection systems may fail to identify complex suspicious "
             "patterns. This system uses machine learning to analyse transaction behaviour, identify anomalies and "
             "flag potentially fraudulent activities.</i>", "callout"),
           P("Translated into requirements, the system must:")]
    st += bullets([
        "<b>Analyse behaviour</b>: judge each transaction against the card's and customer's own history, not just "
        "fixed limits.",
        "<b>Identify anomalies</b>: score how unusual a transaction is, even for patterns not seen before.",
        "<b>Flag fraud</b>: produce a decision with an adjustable trade-off between catching fraud and annoying "
        "customers with false alarms.",
        "<b>Be explainable and usable</b>: analysts must see why something was flagged and be able to score new data.",
        "<b>Be evaluated honestly</b>: tested on later data than it was trained on, as it would be in production.",
    ])

    # ---------------- 3. data
    st += [P("3. Dataset", "h1"),
           P("The data is the public Kaggle dataset <b>computingvictor/transactions-fraud-datasets</b>, the same data "
             "used by the reference Kaggle notebook <i>Financial transactions</i> (jigneshkirtinagda), which explores "
             "it but builds no model. It is a realistic synthetic banking dataset."),
           table([
               ["File", "Content", "Size"],
               ["transactions_data.csv", "date, client, card, amount, channel (swipe / chip / online), merchant, "
                "city, state, ZIP, merchant category code (MCC), error codes", f"{d['n_txn_total']:,} rows"],
               ["cards_data.csv", "brand, type, chip, credit limit, account open date, expiry, PIN change year, "
                "seen on dark web", f"{d['n_cards']:,} rows"],
               ["users_data.csv", "age, gender, location, income, debt, credit score, number of cards",
                f"{d['n_users']:,} rows"],
               ["train_fraud_labels.json", "transaction id to Yes / No fraud label", f"{run['n_labelled']:,} labels"],
               ["mcc_codes.json", "merchant category code descriptions", "109 codes"],
           ], col_widths=[3.6 * cm, 9.4 * cm, 3.0 * cm]),
           Spacer(1, 8),
           P(f"Fraud is rare: <b>{run['fraud_rate']:.3%}</b> of labelled transactions (about 1 in "
             f"{1 / run['fraud_rate']:,.0f}). A model that always says <i>legitimate</i> would be "
             f"{1 - run['fraud_rate']:.2%} accurate and useless, which is why accuracy is not used as a metric."),
           P("Cleaning (as in the reference notebook)", "h2")]
    st += bullets([
        "Money columns are stored as text such as <font face='Courier'>$-77.00</font>; the $ sign is stripped and "
        "values cast to numbers. Negative amounts are refunds.",
        "Dates are parsed; card expiry and account-open dates (MM/YYYY) become proper dates; YES/NO flags become 0/1.",
        "Cards are joined to users by client id and to transactions by card id. Data types are down-cast "
        "(float32, int16, categories) so all 13.3M rows fit in memory, and a cleaned Parquet copy is cached.",
    ])

    # ---------------- 4. EDA
    st += [P("4. Exploratory findings", "h1"),
           figure(fig_amt, 1.0, "Figure 1. Fraudulent transactions tend to be larger, and the fraud rate varies "
                                "strongly by hour of day."),
           P("Fraud patterns shift over time", "h2"),
           P("The most important finding of the analysis is <b>concept drift</b>. Between 2012 and 2016 most "
             "fraud is online. From 2017 onward fraud concentrates on <b>foreign merchants</b>, where roughly one "
             f"transaction in five is fraudulent in 2018-2019 ({drift.loc[2019, 'Foreign merchant']:.1f}% in 2019). "
             "A rule tuned on one period fails on the next, and a model must be trained and tested in time order."),
           figure(fig_drift, 1.0, "Figure 2. Left: fraud rate by channel and year. Right: monthly fraud rate with "
                                  "the train / validation / test boundaries used in this project.")]

    # ---------------- 5. architecture
    st += [P("5. System architecture", "h1"),
           figure(fig_pipe, 1.0, "Figure 3. End-to-end pipeline."),
           P("The pipeline is plain Python (pandas, scikit-learn, LightGBM, SHAP, Streamlit) organised as a small "
             "package:"),
           table([
               ["Component", "File", "Role"],
               ["Data", "src/fraud/data.py", "download, clean, merge, cache"],
               ["Features", "src/fraud/features.py", "transaction, behavioural, card and customer features"],
               ["Models", "src/fraud/models.py", "rules, Isolation Forest, logistic regression, LightGBM, thresholds"],
               ["Evaluation", "src/fraud/evaluate.py", "PR-AUC, ROC-AUC, recall at 1% FPR, confusion counts"],
               ["Explanations", "src/fraud/explain.py", "SHAP reasons with readable feature names"],
               ["Scoring", "src/fraud/scoring.py", "score new raw transactions using each card's history"],
               ["Training", "scripts/train.py", "runs everything end to end, saves models and metrics"],
               ["Analysis", "notebooks/fraud_detection.ipynb", "EDA, model comparison, explainability"],
               ["Dashboard", "app/streamlit_app.py", "overview, alert list, investigation, CSV scoring"],
           ], col_widths=[2.8 * cm, 5.4 * cm, 7.8 * cm])]

    # ---------------- 6. features
    st += [P("6. Feature engineering", "h1"),
           P("Each transaction is described by about 50 features in four groups. The behavioural group is what "
             "lets the model judge a transaction <i>relative to the card's normal behaviour</i>, which rules "
             "cannot do well."),
           table([
               ["Group", "Features", "Why it helps"],
               ["Transaction", "amount, log amount, refund, hour, weekday, night, channel, online, foreign merchant, "
                "merchant category, 7 error-code flags (bad CVV, bad PIN, insufficient balance, ...)",
                "basic risk factors of the payment itself"],
               ["Behaviour (past-only, per card)", "seconds since previous transaction; transactions and spend in "
                "last 1 h / 24 h; amount vs card's average (ratio, z-score); first time at this merchant / "
                "category / state; foreign and online transactions in last 24 h / 7 days; new merchants in last "
                "24 h; past errors; customer-level 24 h velocity",
                "captures bursts, unusual spend and sudden changes in where and how a card is used"],
               ["Card", "seen on dark web, chip, credit limit, amount / limit, cards issued, brand, type",
                "card exposure and capacity"],
               ["Customer", "credit score, income, debt, debt-to-income, income vs local area, number of cards, "
                "gender", "customer context"],
               ["Anomaly", "Isolation Forest score", "unsupervised 'how unusual' signal"],
           ], col_widths=[3.0 * cm, 8.6 * cm, 4.4 * cm]),
           Spacer(1, 8),
           P("No look-ahead (leakage) by design", "h2"),
           P("Every behavioural feature uses only transactions that happened <b>before</b> the current one on the "
             "same card or customer (data sorted by card and time, windows computed with binary search over "
             "timestamps, running sums that exclude the current row). The features can therefore be computed in "
             "exactly the same way when a new transaction arrives. A unit test "
             "(<font face='Courier'>tests/test_features.py</font>) checks this on a hand-made card history."),
           P("Excluded on purpose: time-proxy features", "h2"),
           P("Card age, years since PIN change, customer age, the card's transaction count and card-expired are "
             "computed but <b>not given to the models</b>. They grow steadily with the calendar, so the trees used "
             "them to work out <i>which era</i> a transaction came from (and, for example, that foreign transactions "
             "were safe in 2014) instead of learning fraud behaviour. Removing them makes the model more robust to "
             "drift (section 9).")]

    # ---------------- 7. models
    st += [P("7. Models", "h1"),
           P("Four detectors are trained on the same data and compared on the same unseen test year."),
           P("7.1 Rule-based baseline (what traditional systems do)", "h2"),
           P("A transaction is flagged when at least one rule fires:")]
    st += bullets([f"{name}" for name in models.RULES])
    st += [P("7.2 Isolation Forest (unsupervised anomaly detection)", "h2"),
           P("Learns what typical transactions look like without using labels (200 trees, fitted on a 300,000-row "
             "sample of the training period) and scores how easily each transaction can be isolated. It can react "
             "to new kinds of fraud, and its score is also passed to the supervised models as a feature."),
           P("7.3 Logistic regression", "h2"),
           P("A linear, interpretable baseline: median imputation, standardisation, class-balanced weights, fitted "
             "on all training frauds plus 1.5M sampled legitimate transactions."),
           P("7.4 LightGBM gradient-boosted trees (main model)", "h2"),
           P(f"Learns non-linear combinations of features (for example <i>online + new merchant + several new "
             f"merchants in the last day</i>). Settings: learning rate 0.05, 63 leaves, at least 200 rows per leaf, "
             f"80% row and column sampling, merchant category / channel / card brand / card type / gender treated "
             f"as categorical, early stopping on validation PR-AUC (best: {run['lgbm_best_iteration']} trees). "
             "The fraud class is <b>not</b> up-weighted (see section 9).")]

    # ---------------- 8. evaluation
    st += [P("8. Evaluation method", "h1"),
           P("Time-based split", "h2"),
           table([
               ["Set", "Period", "Labelled transactions", "Used for"],
               ["Train", "Jan 2010 - Jun 2018", f"{split['train']:,}", "fitting all models"],
               ["Validation", "Jul 2018 - Dec 2018", f"{split['valid']:,}",
                "early stopping, choosing thresholds and model settings"],
               ["Test", "Jan 2019 - Oct 2019", f"{split['test']:,} ({n_test_fraud:,} frauds)",
                "final results only, never used for any decision"],
           ], col_widths=[2.2 * cm, 4.0 * cm, 4.2 * cm, 5.6 * cm]),
           Spacer(1, 6),
           P("A random split would let the model see the future and hide drift, so it would overstate performance. "
             "The training period includes the first half of 2018 so that the model has seen the current "
             "foreign-merchant fraud pattern, as a bank retraining on recent data would."),
           P("Metrics", "h2")]
    st += bullets([
        "<b>PR-AUC (average precision)</b>, the headline metric: how well frauds are ranked above legitimate "
        "transactions, focusing on the rare fraud class.",
        "<b>Recall at 1% false-positive rate</b>: share of fraud caught if 1 in 100 legitimate transactions may be "
        "flagged.",
        "<b>Precision, recall, F1 and confusion counts</b> at the chosen threshold.",
        "ROC-AUC is reported but can look high even for weak models when fraud is this rare.",
    ])
    st += [P("Thresholds and risk tiers", "h2"),
           P(f"Thresholds are chosen on the validation set. <b>High risk</b> (score of at least {thr['flag']:.3f}) "
             f"maximises F1 and suits blocking or extra authentication. <b>Medium risk</b> (at least "
             f"{thr['review']:.3f}) is the highest threshold that still reached 80% recall on validation and suits "
             "analyst review. Everything else is <b>Low</b>.")]

    # ---------------- 9. results
    st += [P("9. Results", "h1"),
           P("9.1 Model comparison on the 2019 test year", "h2")]
    rows = [["Model", "PR-AUC", "ROC-AUC", "Recall at 1% FPR", "Precision", "Recall", "F1", "Frauds caught",
             "False alarms"]]
    for name in MODEL_ORDER:
        r = m[name]
        rows.append([f"<b>{name}</b>" if name == "LightGBM" else name, f"{r['pr_auc']:.3f}", f"{r['roc_auc']:.3f}",
                     f"{r['recall_at_1pct_fpr']:.1%}", f"{r['precision']:.1%}", f"{r['recall']:.1%}",
                     f"{r['f1']:.3f}", f"{r['tp']:,}", f"{r['fp']:,}"])
    st += [table(rows, col_widths=[3.3 * cm, 1.75 * cm, 1.9 * cm, 1.8 * cm, 1.75 * cm, 1.5 * cm, 1.3 * cm, 1.75 * cm, 1.95 * cm], highlight_row=4),
           P("Table: precision, recall, F1 and counts use each model's threshold chosen on validation (rules: at "
             "least one rule fires).", "caption"),
           figure(fig_curves, 1.0, "Figure 4. Precision-recall and ROC curves on the test year (ROC x-axis on a "
                                   "log scale to show the low false-positive region that matters in practice).")]
    st += bullets([
        f"<b>LightGBM is clearly best</b>: PR-AUC {lg['pr_auc']:.3f} is {lg['pr_auc'] / rb['pr_auc']:.0f} times the "
        f"rule system and {lg['pr_auc'] / lr['pr_auc']:.1f} times logistic regression.",
        f"<b>Rules</b> do catch all 2019 frauds, but only because one rule (<i>foreign merchant</i>) happens to match "
        f"this year's pattern; they flag {rb['flag_rate']:.1%} of all transactions with {rb['precision']:.1%} "
        "precision. The same rules would have missed the online fraud of 2012-2016.",
        "<b>Isolation Forest alone</b> is weak: unusual is not the same as fraudulent. Its score is still useful as "
        "an input to the supervised model (one of the top features).",
    ])
    rules = d["rules"]
    rrows = [["Rule (2019 test)", "Fires on", "Precision", "Recall"]] + [
        [n, f"{r.iloc[0]:.2%}", f"{r.iloc[1]:.1%}", f"{r.iloc[2]:.1%}"] for n, r in rules.iterrows()]
    st += [P("9.2 Why single rules are not enough", "h2"),
           table(rrows, col_widths=[9.4 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm]),
           Spacer(1, 6),
           P("Each rule either fires on many legitimate transactions or misses most fraud. The model instead "
             "weighs dozens of signals together into one score."),
           P("9.3 Errors and risk tiers", "h2"),
           figure(fig_cm, 1.0, "Figure 5. Left: LightGBM at the High-risk threshold. Right: how the three risk "
                               "tiers split the 2019 transactions."),
           P(f"Of {len(sc):,} test transactions, {int(tiers.loc['High', 'n']):,} are High risk "
             f"({tiers.loc['High', 'fraud'] / tiers.loc['High', 'n']:.0%} of them fraud) and "
             f"{int(tiers.loc['Medium', 'n']):,} Medium ({tiers.loc['Medium', 'fraud'] / tiers.loc['Medium', 'n']:.0%} "
             f"fraud). The Low tier still contains {int(tiers.loc['Low', 'fraud']):,} frauds "
             f"({tiers.loc['Low', 'fraud'] / n_test_fraud:.0%} of all fraud): these look like normal activity for "
             "the card and are the main target for future work.")]

    st += [P("9.4 What made the difference during development", "h2"),
           P("The first version of LightGBM scored a PR-AUC of only 0.48 on test and its ROC-AUC (0.87) was lower "
             "than logistic regression's. Analysis showed that about 12% of frauds, all at foreign merchants, "
             "scored below an ordinary legitimate transaction. Variants were then compared <b>on the validation "
             "set only</b>:"),
           table([
               ["LightGBM variant", "Validation PR-AUC", "Validation ROC-AUC"],
               ["Fraud class up-weighted 10x, all features (first version)", "0.594", "0.866"],
               ["Up-weighted, time-proxy features removed", "0.591", "0.793"],
               ["Up-weighted, recent data weighted more", "0.631", "0.840"],
               ["No up-weighting, all features", "0.837", "0.999"],
               ["<b>No up-weighting, time-proxy features removed (final)</b>", "<b>0.844</b>", "<b>0.999</b>"],
           ], col_widths=[10 * cm, 3 * cm, 3 * cm], highlight_row=5),
           P("Values from the development experiments on the full data.", "caption")]
    st += bullets([
        "<b>Up-weighting the rare class hurt</b>. It is a common recipe for imbalanced data, but here it made the "
        "trees over-react to small groups of training frauds and rank whole segments wrongly. Gradient boosting "
        "already handles the imbalance when the goal is ranking.",
        "<b>Removing time-proxy features</b> gave a further small gain and makes the model rely on behaviour rather "
        "than on which year a transaction came from.",
    ])

    # ---------------- 10. explainability
    st += [P("10. Explainability", "h1"),
           figure(fig_imp, 0.85, "Figure 6. Features the model relies on most."),
           P("The strongest signals are the payment channel, a burst of <b>new merchants on the card in the last "
             "24 hours</b>, merchant category, the Isolation Forest anomaly score, past errors on the card, the hour "
             "of day, and online / foreign activity: a mix of behavioural and transaction features that no single "
             "rule captures."),
           P("Explaining a single alert", "h2"),
           P(f"For each flagged transaction, SHAP values show how much each feature pushed the score up or down. "
             f"Example: transaction {int(case['id'])} on {case['date']:%d %b %Y %H:%M}, ${case['amount']:,.2f} at "
             f"{case['merchant_city']} ({case['mcc_description']}), scored {case['fraud_score']:.3f} and confirmed as "
             "fraud."),
           figure(fig_ex, 0.85, "Figure 7. Top reasons for one High-risk transaction.")]

    # ---------------- 11. application
    st += [P("11. The application", "h1"),
           P("A Streamlit dashboard (<font face='Courier'>streamlit run app/streamlit_app.py</font>) puts the "
             "system in front of analysts:"),
           table([
               ["Page", "What it does"],
               ["Overview", "headline metrics, model comparison table, PR and ROC curves, fraud rate over time with "
                "the split boundaries, top features, per-rule statistics"],
               ["Flagged transactions", "filterable list of scored 2019 transactions (risk tier, date range, amount, "
                "merchant category), precision of the current selection, amount at risk, CSV download"],
               ["Investigate a transaction", "score and risk tier, transaction details, SHAP 'why this score' chart, "
                "the card's activity in the surrounding days"],
               ["Score new data", "upload a CSV in the original transactions format (or try a random sample); "
                "returns score, risk tier and top reasons per transaction, downloadable"],
           ], col_widths=[4 * cm, 12 * cm]),
           Spacer(1, 8),
           P("Scoring new transactions", "h2"),
           P("Behavioural features need the card's history. When a file is uploaded, the system loads each card's "
             "earlier transactions from the cleaned dataset, appends the new rows, recomputes the features exactly "
             "as in training, adds the Isolation Forest score and applies LightGBM. Re-scoring known 2019 "
             "transactions this way reproduces the training-time scores (largest difference 0.0002)."),

           P("12. How to run", "h1"),
           Paragraph(
               "python3.11 -m venv .venv &amp;&amp; source .venv/bin/activate<br/>"
               "pip install -r requirements.txt<br/>"
               "python scripts/download_data.py&nbsp;&nbsp;&nbsp;# public Kaggle data, ~350 MB<br/>"
               "python scripts/train.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# ~4 min; "
               "--sample 500000 for a quick run<br/>"
               "pytest -q&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
               "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# leakage test<br/>"
               "jupyter notebook notebooks/fraud_detection.ipynb<br/>"
               "streamlit run app/streamlit_app.py<br/>"
               "python scripts/build_report.py&nbsp;&nbsp;&nbsp;&nbsp;# this report", S["code"]),
           P("Training on all 13.3 million transactions takes about 3-4 minutes on a 16 GB laptop. Outputs go to "
             "<font face='Courier'>models/</font> (models, metrics, curves, thresholds) and "
             "<font face='Courier'>data/processed/</font> (features and scored test set)."),

           P("13. Limitations and future work", "h1")]
    st += bullets([
        "<b>Synthetic data.</b> The dataset is generated, and its fraud regimes switch more sharply than real "
        "fraud usually does. Results show the method works; real-world numbers would need real data.",
        "<b>Drift needs monitoring.</b> Fraud patterns change, so the model should be retrained regularly and its "
        "score distribution and alert precision tracked over time.",
        "<b>Missed fraud.</b> About a third of 2019 fraud stays in the Low tier because it looks like normal card "
        "activity. Sequence models over each card's history, or merchant-level features, could find more of it.",
        "<b>Cost-sensitive decisions.</b> Thresholds currently balance counts; weighting by transaction amount "
        "and the cost of a false alarm would align them with money saved.",
        "<b>Real-time serving.</b> The scoring code is ready for batch files; a streaming version would keep "
        "per-card running statistics in a fast store instead of re-reading history.",
    ])
    st += [P("14. Conclusion", "h1"),
           P(f"The project meets the problem statement. It analyses each transaction against the card's own "
             f"behaviour, scores anomalies, and flags likely fraud with clear reasons. Compared with a rule-based "
             f"system it ranks fraud about {lg['pr_auc'] / rb['pr_auc']:.0f} times better (PR-AUC) and cuts false "
             f"alarms from {rb['fp']:,} to {lg['fp']:,} on a full year of unseen transactions. Evaluating in time "
             "order also showed why static rules fail: the fraud itself keeps changing.")]

    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(2 * cm, 1.2 * cm, "Fraud Detection in Digital Financial Transactions")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
        canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title="Fraud Detection in Digital Financial Transactions",
                            subject="Machine-learning fraud detection project report")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    build()
