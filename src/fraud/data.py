"""Loading and cleaning of the Kaggle `transactions-fraud-datasets` files.

Cleaning mirrors the reference Kaggle EDA notebook (strip `$` from money columns, parse dates,
merge cards -> users -> transactions on client_id / card_id), with dtype downcasting so the
13M-row transaction table fits comfortably in memory.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def download_dataset(dest: Path = config.RAW_DIR) -> Path:
    """Download the Kaggle dataset into `dest` (works anonymously for this public dataset)."""
    import kagglehub

    dest.mkdir(parents=True, exist_ok=True)
    src = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    for f in src.iterdir():
        target = dest / f.name
        if not target.exists():
            shutil.copy2(f, target)
    return dest


def _money(s: pd.Series, dtype="float32") -> pd.Series:
    return s.astype(str).str.replace("$", "", regex=False).astype(float).astype(dtype)


def load_transactions(path: Path = config.TRANSACTIONS_CSV, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={
            "id": "int64", "client_id": "int32", "card_id": "int32", "merchant_id": "int32",
            "use_chip": "category", "merchant_city": "category", "merchant_state": "category",
            "zip": "float32", "mcc": "int16", "errors": "category",
        },
    )
    return clean_transactions(df)


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning to a raw transactions frame (also used for uploaded CSVs)."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df["amount"]):
        df["amount"] = _money(df["amount"])
    df["amount"] = df["amount"].astype("float32")
    df["date"] = pd.to_datetime(df["date"])
    for c in ["use_chip", "merchant_city", "merchant_state", "errors"]:
        if c in df:
            df[c] = df[c].astype("category")
    return df


def load_cards(path: Path = config.CARDS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["credit_limit"] = _money(df["credit_limit"])
    df["expires"] = pd.to_datetime(df["expires"], format="%m/%Y")
    df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], format="%m/%Y")
    df["has_chip"] = (df["has_chip"].str.upper() == "YES").astype("int8")
    df["card_on_dark_web"] = (df["card_on_dark_web"].str.upper() == "YES").astype("int8")
    return df.rename(columns={"id": "card_id"})


def load_users(path: Path = config.USERS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["per_capita_income", "yearly_income", "total_debt"]:
        df[c] = _money(df[c])
    return df.rename(columns={"id": "client_id"})


def load_mcc(path: Path = config.MCC_JSON) -> dict[int, str]:
    with open(path) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_labels(path: Path = config.LABELS_JSON) -> pd.Series:
    """Return a Series indexed by transaction id with 1 = fraud, 0 = legitimate."""
    with open(path) as f:
        target = json.load(f)["target"]
    ids = np.fromiter((int(k) for k in target.keys()), dtype="int64", count=len(target))
    vals = np.fromiter((v == "Yes" for v in target.values()), dtype="int8", count=len(target))
    return pd.Series(vals, index=pd.Index(ids, name="id"), name="is_fraud")


def load_clean_transactions(card_ids=None) -> pd.DataFrame:
    """Load the cached cleaned transactions (building the cache from CSV on first use)."""
    if not config.CLEAN_TXN_PARQUET.exists():
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        load_transactions().to_parquet(config.CLEAN_TXN_PARQUET, index=False)
    filters = [("card_id", "in", list(map(int, card_ids)))] if card_ids is not None else None
    df = pd.read_parquet(config.CLEAN_TXN_PARQUET, filters=filters)
    for c in ["use_chip", "merchant_city", "merchant_state", "errors"]:
        df[c] = df[c].astype("category")
    return df
