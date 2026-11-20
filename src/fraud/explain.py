"""SHAP explanations: why did the model flag a transaction?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config

FEATURE_LABELS = {
    "amount": "Amount", "log_abs_amount": "Amount (log)", "is_refund": "Refund",
    "hour": "Hour of day", "day_of_week": "Day of week", "is_night": "Night-time",
    "is_online": "Online purchase", "is_foreign": "Foreign merchant", "has_error": "Had an error",
    "n_errors": "Number of errors", "err_bad_cvv": "Bad CVV", "err_bad_card_number": "Bad card number",
    "err_bad_expiration": "Bad expiration", "err_bad_pin": "Bad PIN", "err_bad_zipcode": "Bad zipcode",
    "err_insufficient_balance": "Insufficient balance", "err_technical_glitch": "Technical glitch",
    "secs_since_prev_txn": "Seconds since card's previous txn", "txn_count_1h": "Card txns in last hour",
    "txn_count_24h": "Card txns in last 24h", "amount_sum_24h": "Card spend in last 24h",
    "card_txn_number": "Card's past txn count", "card_prev_mean_amount": "Card's usual amount",
    "amount_zscore_card": "Amount vs card's usual (z-score)", "amount_ratio_card_mean": "Amount / card's usual",
    "new_merchant_for_card": "First time at this merchant", "new_mcc_for_card": "First time in this category",
    "new_state_for_card": "First time in this state", "card_prev_error_count": "Card's past errors",
    "client_txn_count_24h": "Client txns in last 24h",
    "card_foreign_count_24h": "Card's foreign txns in last 24h", "card_foreign_count_7d": "Card's foreign txns in last 7d",
    "card_online_count_24h": "Card's online txns in last 24h",
    "card_new_merchant_count_24h": "Card's new-merchant txns in last 24h", "card_on_dark_web": "Card on dark web",
    "has_chip": "Card has chip", "credit_limit": "Credit limit", "amount_to_limit": "Amount / credit limit",
    "num_cards_issued": "Cards issued", "card_age_years": "Card age (years)",
    "years_since_pin_change": "Years since PIN change", "card_expired": "Card expired",
    "age_at_txn": "Customer age", "credit_score": "Credit score", "yearly_income": "Yearly income",
    "total_debt": "Total debt", "debt_to_income": "Debt / income", "num_credit_cards": "Customer's cards",
    "income_to_area": "Income / area income", "use_chip": "Channel (swipe/chip/online)",
    "mcc": "Merchant category", "card_brand": "Card brand", "card_type": "Card type", "gender": "Gender",
    config.ANOMALY_FEATURE: "Anomaly score (Isolation Forest)",
}


def explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.booster_ if hasattr(model, "booster_") else model)


def shap_values(expl: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    sv = expl.shap_values(X)
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    return np.asarray(sv)


def top_reasons(expl, X_row: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Top-k features pushing this transaction's score up or down (log-odds contribution)."""
    sv = shap_values(expl, X_row)[0]
    df = pd.DataFrame({
        "feature": X_row.columns,
        "label": [FEATURE_LABELS.get(c, c) for c in X_row.columns],
        "value": X_row.iloc[0].to_numpy(),
        "contribution": sv,
    })
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)
