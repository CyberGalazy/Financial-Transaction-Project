"""Feature engineering.

Three groups of features describe each transaction:
* transaction attributes (amount, time, channel, merchant location, error codes)
* behaviour of the card/client *before* this transaction (velocity, deviation from usual spend,
  first-time merchant/state, ...). Every behavioural feature looks only at earlier transactions,
  so it can be computed the same way at scoring time and does not leak the future.
* static card and user profile attributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

USE_CHIP_CODES = {"Swipe Transaction": 0, "Chip Transaction": 1, "Online Transaction": 2}
CARD_BRAND_CODES = {"Visa": 0, "Mastercard": 1, "Amex": 2, "Discover": 3}
CARD_TYPE_CODES = {"Debit": 0, "Credit": 1, "Debit (Prepaid)": 2}
GENDER_CODES = {"Female": 0, "Male": 1}


def _map_codes(s: pd.Series, mapping: dict) -> pd.Series:
    return s.astype(object).map(mapping).fillna(-1).astype("int16")


def _error_flags(errors: pd.Series) -> pd.DataFrame:
    """Multi-hot encode the comma-separated `errors` column (computed per category for speed)."""
    errors = errors.astype("category")
    cats = pd.Series(errors.cat.categories.astype(str))
    codes = errors.cat.codes.to_numpy()
    out = {}
    for e in config.ERROR_TYPES:
        per_cat = np.append(cats.str.contains(e, regex=False).to_numpy(), False)  # code -1 -> False
        out[f"err_{e.lower().replace(' ', '_')}"] = per_cat[codes].astype("int8")
    flags = pd.DataFrame(out, index=errors.index)
    flags["n_errors"] = flags.sum(axis=1).astype("int8")
    flags["has_error"] = (flags["n_errors"] > 0).astype("int8")
    return flags


def add_transaction_features(df: pd.DataFrame, mcc_map: dict[int, str]) -> pd.DataFrame:
    amt = df["amount"].astype("float32")
    df["log_abs_amount"] = np.log1p(np.abs(amt)).astype("float32")
    df["is_refund"] = (amt < 0).astype("int8")
    df["hour"] = df["date"].dt.hour.astype("int8")
    df["day_of_week"] = df["date"].dt.day_of_week.astype("int8")
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 23)).astype("int8")
    city = df["merchant_city"].astype(str)
    state = df["merchant_state"].astype(object)
    df["is_online"] = (city == "ONLINE").astype("int8")
    df["is_foreign"] = (state.notna() & ~state.isin(config.US_STATES) & (df["is_online"] == 0)).astype("int8")
    df["mcc_description"] = df["mcc"].astype(int).map(mcc_map).astype(object).fillna("Unknown").astype("category")
    flags = _error_flags(df["errors"])
    for c in flags:
        df[c] = flags[c]
    return df


def _window_left(group_key: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    """For rows sorted by (group, time): index of the first row of the same group within `window` seconds."""
    key = group_key.astype("int64") * 10**10 + t
    return np.searchsorted(key, key - window, side="left")


def _window_sum(values: np.ndarray, left: np.ndarray) -> np.ndarray:
    """Sum of `values` over earlier rows [left, i) (excludes the current row)."""
    csum = np.concatenate([[0.0], np.cumsum(values, dtype="float64")])
    return csum[np.arange(len(values))] - csum[left]


def _window_counts(group_key: np.ndarray, t: np.ndarray, amount: np.ndarray, window: int):
    """Count / amount-sum of earlier rows of the same group within `window` seconds."""
    left = _window_left(group_key, t, window)
    return np.arange(len(left)) - left, _window_sum(amount, left)


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only per-card / per-client behaviour. Returns the frame sorted by card and time."""
    df = df.sort_values(["card_id", "date", "id"], kind="mergesort").reset_index(drop=True)
    t = df["date"].to_numpy().astype("datetime64[s]").astype("int64")
    card = df["card_id"].to_numpy()
    amt = df["amount"].to_numpy().astype("float64")
    g = df.groupby("card_id", sort=False)

    n_prev = g.cumcount().to_numpy()
    df["card_txn_number"] = n_prev.astype("int32")
    df["secs_since_prev_txn"] = g["date"].diff().dt.total_seconds().astype("float32")

    c1h, _ = _window_counts(card, t, amt, 3600)
    c24, s24 = _window_counts(card, t, amt, 86400)
    df["txn_count_1h"] = c1h.astype("int16")
    df["txn_count_24h"] = c24.astype("int16")
    df["amount_sum_24h"] = s24.astype("float32")

    prev_sum = g["amount"].cumsum().to_numpy().astype("float64") - amt
    sq = pd.Series(amt * amt, index=df.index)
    prev_sumsq = sq.groupby(df["card_id"], sort=False).cumsum().to_numpy() - amt * amt
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n_prev > 0, prev_sum / n_prev, np.nan)
        var = np.where(n_prev > 1, prev_sumsq / n_prev - mean * mean, np.nan)
        std = np.sqrt(np.clip(var, 0, None))
        df["card_prev_mean_amount"] = mean.astype("float32")
        df["amount_zscore_card"] = ((amt - mean) / (std + 1.0)).astype("float32")
        df["amount_ratio_card_mean"] = (amt / (np.abs(mean) + 1.0)).astype("float32")

    df["new_merchant_for_card"] = (df.groupby(["card_id", "merchant_id"], sort=False).cumcount() == 0).astype("int8")
    df["new_mcc_for_card"] = (df.groupby(["card_id", "mcc"], sort=False).cumcount() == 0).astype("int8")
    state_code = df["merchant_state"].cat.add_categories(["__ONLINE__"]).fillna("__ONLINE__").cat.codes
    df["new_state_for_card"] = (
        pd.DataFrame({"c": df["card_id"], "s": state_code}).groupby(["c", "s"], sort=False).cumcount() == 0
    ).astype("int8")

    left24 = _window_left(card, t, 86400)
    left7d = _window_left(card, t, 7 * 86400)
    foreign = df["is_foreign"].to_numpy().astype("float64")
    df["card_foreign_count_24h"] = _window_sum(foreign, left24).astype("int16")
    df["card_foreign_count_7d"] = _window_sum(foreign, left7d).astype("int16")
    df["card_online_count_24h"] = _window_sum(df["is_online"].to_numpy().astype("float64"), left24).astype("int16")
    df["card_new_merchant_count_24h"] = _window_sum(
        df["new_merchant_for_card"].to_numpy().astype("float64"), left24).astype("int16")

    df["card_prev_error_count"] = (g["has_error"].cumsum() - df["has_error"]).astype("int32")

    # Client-level velocity (a client may own several cards).
    client = df["client_id"].to_numpy()
    order = np.lexsort((df["id"].to_numpy(), t, client))
    cc, _ = _window_counts(client[order], t[order], amt[order], 86400)
    client_cnt = np.empty_like(cc)
    client_cnt[order] = cc
    df["client_txn_count_24h"] = client_cnt.astype("int16")
    return df


def add_profile_features(df: pd.DataFrame, cards: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Join card and user attributes (via index lookup, cheaper than a merge on 13M rows)."""
    c = cards.set_index("card_id")
    u = users.set_index("client_id")
    cid = df["card_id"].to_numpy()
    uid = df["client_id"].to_numpy()

    for col in ["card_on_dark_web", "has_chip", "credit_limit", "num_cards_issued", "year_pin_last_changed"]:
        df[col] = c[col].reindex(cid).to_numpy()
    df["card_brand"] = _map_codes(pd.Series(c["card_brand"].reindex(cid).to_numpy()), CARD_BRAND_CODES).to_numpy()
    df["card_type"] = _map_codes(pd.Series(c["card_type"].reindex(cid).to_numpy()), CARD_TYPE_CODES).to_numpy()
    expires = pd.to_datetime(c["expires"].reindex(cid).to_numpy())
    opened = pd.to_datetime(c["acct_open_date"].reindex(cid).to_numpy())
    dates = df["date"].to_numpy()
    df["card_age_years"] = ((dates - opened.to_numpy()) / np.timedelta64(1, "D") / 365.25).astype("float32")
    # card valid through the end of its expiry month
    df["card_expired"] = (dates >= (expires + pd.offsets.MonthBegin(1)).to_numpy()).astype("int8")
    df["years_since_pin_change"] = (df["date"].dt.year - df["year_pin_last_changed"]).astype("float32")
    df["amount_to_limit"] = (df["amount"] / (df["credit_limit"] + 1.0)).astype("float32")

    for col in ["credit_score", "yearly_income", "total_debt", "num_credit_cards", "per_capita_income", "birth_year"]:
        df[col] = u[col].reindex(uid).to_numpy()
    df["gender"] = _map_codes(pd.Series(u["gender"].reindex(uid).to_numpy()), GENDER_CODES).to_numpy()
    df["age_at_txn"] = (df["date"].dt.year - df["birth_year"]).astype("float32")
    df["debt_to_income"] = (df["total_debt"] / (df["yearly_income"] + 1.0)).astype("float32")
    df["income_to_area"] = (df["yearly_income"] / (df["per_capita_income"] + 1.0)).astype("float32")
    df = df.drop(columns=["year_pin_last_changed", "per_capita_income", "birth_year"])
    return df


def build_features(txn: pd.DataFrame, cards: pd.DataFrame, users: pd.DataFrame,
                   mcc_map: dict[int, str]) -> pd.DataFrame:
    """Full feature table for cleaned transactions (sorted by card, time)."""
    df = add_transaction_features(txn.copy(), mcc_map)
    df = add_behavioral_features(df)
    df = add_profile_features(df, cards, users)
    df["use_chip_code"] = _map_codes(df["use_chip"], USE_CHIP_CODES)
    return df


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix in the column order models expect (categoricals as integer codes)."""
    X = df[config.NUMERIC_FEATURES].astype("float32")
    X["use_chip"] = df["use_chip_code"].astype("int16") if "use_chip_code" in df else _map_codes(df["use_chip"], USE_CHIP_CODES)
    X["mcc"] = df["mcc"].astype("int16")
    for c in ["card_brand", "card_type", "gender"]:
        X[c] = df[c].astype("int16")
    if config.ANOMALY_FEATURE in df:
        X[config.ANOMALY_FEATURE] = df[config.ANOMALY_FEATURE].astype("float32")
    return X
