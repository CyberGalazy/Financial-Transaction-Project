"""End-to-end training: load -> features -> time split -> train 4 detectors -> evaluate -> save.

Usage:
    python scripts/train.py                  # full dataset (~13M transactions)
    python scripts/train.py --sample 500000  # quick run on a subset of cards
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud import config, data, evaluate, features, models  # noqa: E402


def log(msg: str, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def sample_cards(txn: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Keep whole card histories (row sampling would break behavioural features)."""
    counts = txn["card_id"].value_counts().sample(frac=1.0, random_state=seed)
    keep = counts.index[counts.cumsum() <= n_rows]
    return txn[txn["card_id"].isin(keep)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="approx. number of transactions to use")
    args = ap.parse_args()

    if not config.TRANSACTIONS_CSV.exists():
        log("Downloading dataset from Kaggle ...")
        data.download_dataset()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log("Loading data ...")
    txn = data.load_clean_transactions()
    cards, users, mcc_map = data.load_cards(), data.load_users(), data.load_mcc()
    labels = data.load_labels()
    if args.sample:
        txn = sample_cards(txn, args.sample, config.RANDOM_STATE)
    log(f"{len(txn):,} transactions, {len(labels):,} labels")

    log("Building features ...")
    df = features.build_features(txn, cards, users, mcc_map)
    del txn
    gc.collect()
    df["is_fraud"] = labels.reindex(df["id"].to_numpy()).to_numpy()
    del labels

    monthly = (df.dropna(subset=["is_fraud"]).set_index("date")
               .resample("MS").agg(transactions=("id", "size"), frauds=("is_fraud", "sum"),
                                   amount=("amount", "sum")))
    monthly["fraud_rate"] = monthly["frauds"] / monthly["transactions"]
    monthly.to_csv(config.MODELS_DIR / "fraud_by_month.csv")

    df = df[df["is_fraud"].notna()].reset_index(drop=True)
    df["is_fraud"] = df["is_fraud"].astype("int8")
    log(f"{len(df):,} labelled transactions, fraud rate {df['is_fraud'].mean():.4%}")

    split = np.where(df["date"] < config.TRAIN_END, "train",
                     np.where(df["date"] < config.VALID_END, "valid", "test"))
    df["split"] = pd.Categorical(split, categories=["train", "valid", "test"])
    for s in ["train", "valid", "test"]:
        part = df[df["split"] == s]
        log(f"  {s:5s}: {len(part):>10,} rows  {part['date'].min():%Y-%m-%d} .. {part['date'].max():%Y-%m-%d}"
            f"  frauds={int(part['is_fraud'].sum()):,}")
    assert df.loc[df["split"] == "train", "date"].max() < df.loc[df["split"] == "valid", "date"].min()
    assert df.loc[df["split"] == "valid", "date"].max() < df.loc[df["split"] == "test", "date"].min()

    is_train = (df["split"] == "train").to_numpy()
    is_valid = (df["split"] == "valid").to_numpy()
    is_test = (df["split"] == "test").to_numpy()
    y = df["is_fraud"]

    # 1. Isolation Forest -> anomaly score (also a feature for the supervised models)
    log("Training Isolation Forest ...")
    X = features.model_matrix(df)
    iforest = models.train_iforest(X[is_train])
    df[config.ANOMALY_FEATURE] = models.iforest_score(iforest, X)
    X[config.ANOMALY_FEATURE] = df[config.ANOMALY_FEATURE].to_numpy()

    scores = {
        "Rule-based baseline": models.rule_score(df),
        "Isolation Forest": df[config.ANOMALY_FEATURE].to_numpy(),
    }

    log("Training Logistic Regression ...")
    logreg = models.train_logreg(X[is_train], y[is_train])
    scores["Logistic Regression"] = models.logreg_score(logreg, X)

    log("Training LightGBM ...")
    lgbm = models.train_lgbm(X[is_train], y[is_train], X[is_valid], y[is_valid])
    log(f"  best iteration: {lgbm.best_iteration_}")
    scores["LightGBM"] = models.lgbm_score(lgbm, X)

    # 2. Thresholds on validation, metrics on the untouched test year
    log("Evaluating ...")
    metrics, curves, thresholds = {}, {}, {}
    yv, yt = y[is_valid].to_numpy(), y[is_test].to_numpy()
    for name, s in scores.items():
        if name == "Rule-based baseline":
            thr = {"flag": 1.0, "review": 1.0}   # flagged when at least one rule fires
        else:
            thr = models.choose_thresholds(yv, s[is_valid])
        thresholds[name] = thr
        metrics[name] = evaluate.evaluate(yt, s[is_test], thr["flag"])
        curves[name] = evaluate.curves(yt, s[is_test])

    table = pd.DataFrame(metrics).T[["pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "f1",
                                     "flag_rate", "tp", "fp", "fn"]]
    print("\nTest-set results (2019, unseen):\n" + table.to_string(float_format=lambda v: f"{v:.4f}") + "\n")

    test = df[is_test]
    rule_stats = models.rule_hits(test).assign(is_fraud=test["is_fraud"].to_numpy())
    rule_table = pd.DataFrame({
        "fires_on_pct_of_txns": rule_stats.drop(columns="is_fraud").mean(),
        "precision": rule_stats.drop(columns="is_fraud").apply(
            lambda c: rule_stats.loc[c == 1, "is_fraud"].mean() if c.sum() else 0.0),
        "recall": rule_stats.drop(columns="is_fraud").apply(
            lambda c: c[rule_stats["is_fraud"] == 1].mean()),
    })
    rule_table.to_csv(config.MODELS_DIR / "rule_stats.csv")

    # 3. Save artifacts
    log("Saving artifacts ...")
    joblib.dump(iforest, config.MODELS_DIR / "iforest.joblib")
    joblib.dump(logreg, config.MODELS_DIR / "logreg.joblib")
    joblib.dump(lgbm, config.MODELS_DIR / "lgbm.joblib")
    (config.MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (config.MODELS_DIR / "curves.json").write_text(json.dumps(curves))
    (config.MODELS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (config.MODELS_DIR / "run_info.json").write_text(json.dumps({
        "sample": args.sample,
        "n_labelled": int(len(df)),
        "split_sizes": df["split"].value_counts().to_dict(),
        "fraud_rate": float(y.mean()),
        "lgbm_best_iteration": int(lgbm.best_iteration_ or 0),
    }, indent=2))
    imp = pd.DataFrame({
        "feature": lgbm.booster_.feature_name(),
        "gain": lgbm.booster_.feature_importance("gain"),
        "split": lgbm.booster_.feature_importance("split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(config.MODELS_DIR / "feature_importance.csv", index=False)

    keep = list(dict.fromkeys(config.DISPLAY_COLUMNS + config.NUMERIC_FEATURES + [
        "use_chip_code", "card_brand", "card_type", "gender", config.ANOMALY_FEATURE, "is_fraud"]))
    scored = df.loc[is_test, keep].copy()
    scored["rule_score"] = scores["Rule-based baseline"][is_test]
    scored["logreg_score"] = scores["Logistic Regression"][is_test]
    scored["fraud_score"] = scores["LightGBM"][is_test]
    scored["risk_level"] = models.risk_level(scored["fraud_score"].to_numpy(), thresholds["LightGBM"])
    scored["errors"] = scored["errors"].astype(str).replace("nan", "")
    scored.sort_values("date").to_parquet(config.SCORED_TEST_PARQUET, index=False)

    df.drop(columns=["split"]).assign(split=df["split"].astype(str)).to_parquet(config.FEATURES_PARQUET, index=False)
    log("Done.")


if __name__ == "__main__":
    main()
