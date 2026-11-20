"""Models: a rule-based baseline, Isolation Forest (unsupervised anomalies),
Logistic Regression and LightGBM (main supervised model)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# ---------------------------------------------------------------- rule-based baseline
RULES = {
    "Amount > 5x card's usual spend (and > $200)":
        lambda d: (d["amount"] > 5 * d["card_prev_mean_amount"].abs().fillna(np.inf)) & (d["amount"] > 200),
    "Card seen on dark web": lambda d: d["card_on_dark_web"] == 1,
    "Foreign merchant": lambda d: d["is_foreign"] == 1,
    "Security error (bad CVV / PIN / card no. / expiry)":
        lambda d: (d["err_bad_cvv"] + d["err_bad_pin"] + d["err_bad_card_number"] + d["err_bad_expiration"]) > 0,
    "3+ transactions on card within 1 hour": lambda d: d["txn_count_1h"] >= 3,
    "New state for card and amount > $300": lambda d: (d["new_state_for_card"] == 1) & (d["amount"] > 300),
}


def rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: fn(df).astype("int8") for name, fn in RULES.items()}, index=df.index)


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """Number of rules fired; a transaction is flagged by the rule system when >= 1."""
    return rule_hits(df).sum(axis=1).to_numpy().astype("float32")


# ---------------------------------------------------------------- isolation forest
IF_FEATURES = config.NUMERIC_FEATURES


def train_iforest(X: pd.DataFrame, sample_size: int = 300_000, seed: int = config.RANDOM_STATE):
    Xs = X[IF_FEATURES].sample(min(sample_size, len(X)), random_state=seed)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=200, max_samples=512, random_state=seed, n_jobs=-1),
    )
    model.fit(Xs)
    return model


def iforest_score(model, X: pd.DataFrame, batch: int = 1_000_000) -> np.ndarray:
    """Anomaly score: higher = more unusual."""
    out = [-model.score_samples(X[IF_FEATURES].iloc[i:i + batch]) for i in range(0, len(X), batch)]
    return np.concatenate(out).astype("float32") if out else np.array([], dtype="float32")


# ---------------------------------------------------------------- logistic regression
LOGREG_FEATURES = config.NUMERIC_FEATURES + [config.ANOMALY_FEATURE]


def train_logreg(X: pd.DataFrame, y: pd.Series, max_legit: int = 1_500_000, seed: int = config.RANDOM_STATE):
    """Fit on all frauds + a random sample of legitimate transactions (class-balanced)."""
    fraud_idx = y.index[y == 1]
    legit_idx = y.index[y == 0]
    if len(legit_idx) > max_legit:
        legit_idx = pd.Index(np.random.default_rng(seed).choice(legit_idx, max_legit, replace=False))
    idx = fraud_idx.append(legit_idx)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000),
    )
    model.fit(X.loc[idx, LOGREG_FEATURES], y.loc[idx])
    return model


def logreg_score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[LOGREG_FEATURES])[:, 1].astype("float32")


# ---------------------------------------------------------------- lightgbm
LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    scale_pos_weight=1.0,   # up-weighting frauds hurt ranking badly on validation (AP 0.59 vs 0.84)
    max_cat_to_onehot=8,
    cat_smooth=20,
    n_jobs=-1,
    verbose=-1,
)


def train_lgbm(X_train, y_train, X_valid, y_valid, seed: int = config.RANDOM_STATE, params: dict | None = None):
    model = lgb.LGBMClassifier(**{**LGBM_PARAMS, **(params or {})}, random_state=seed)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="average_precision",
        categorical_feature=config.CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(100, first_metric_only=False, verbose=False), lgb.log_evaluation(200)],
    )
    return model


def lgbm_score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[config.MODEL_FEATURES])[:, 1].astype("float32")


# ---------------------------------------------------------------- thresholds
def choose_thresholds(y: np.ndarray, score: np.ndarray, target_recall: float = 0.80) -> dict:
    """`flag` = threshold maximising F1; `review` = highest threshold that still reaches target recall."""
    precision, recall, thr = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.nanargmax(f1))
    ok = np.where(recall >= target_recall)[0]
    review = float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {
        "flag": float(thr[best]),
        "review": min(review, float(thr[best])),
        "valid_f1_at_flag": float(f1[best]),
        "target_recall": target_recall,
    }


def risk_level(score: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(score >= thresholds["flag"], "High",
                    np.where(score >= thresholds["review"], "Medium", "Low"))
