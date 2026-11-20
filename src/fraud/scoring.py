"""Score new transactions with the trained models (used by the Streamlit app and notebook)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from . import config, data, explain, features, models

REQUIRED_COLUMNS = ["id", "date", "client_id", "card_id", "amount", "use_chip", "merchant_id",
                    "merchant_city", "merchant_state", "zip", "mcc", "errors"]


@dataclass
class Artifacts:
    lgbm: object
    iforest: object
    thresholds: dict
    cards: pd.DataFrame
    users: pd.DataFrame
    mcc_map: dict

    @classmethod
    def load(cls) -> "Artifacts":
        return cls(
            lgbm=joblib.load(config.MODELS_DIR / "lgbm.joblib"),
            iforest=joblib.load(config.MODELS_DIR / "iforest.joblib"),
            thresholds=json.loads((config.MODELS_DIR / "thresholds.json").read_text())["LightGBM"],
            cards=data.load_cards(),
            users=data.load_users(),
            mcc_map=data.load_mcc(),
        )


def score_transactions(new: pd.DataFrame, art: Artifacts, with_reasons: bool = True) -> pd.DataFrame:
    """Score raw transactions (same schema as transactions_data.csv).

    Each card's earlier history is loaded from the cleaned dataset so behavioural features
    (velocity, usual spend, first-time merchant, ...) are computed exactly as in training.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in new.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    new = data.clean_transactions(new[REQUIRED_COLUMNS])
    new["mcc"] = new["mcc"].astype("int16")

    history = data.load_clean_transactions(card_ids=new["card_id"].unique())
    history = history[(history["date"] <= new["date"].max()) & ~history["id"].isin(new["id"])]
    combined = pd.concat([history, new], ignore_index=True)
    for c in ["use_chip", "merchant_city", "merchant_state", "errors"]:
        combined[c] = combined[c].astype(str).replace({"nan": np.nan, "None": np.nan}).astype("category")

    feats = features.build_features(combined, art.cards, art.users, art.mcc_map)
    feats = feats[feats["id"].isin(new["id"])].reset_index(drop=True)
    X = features.model_matrix(feats)
    X[config.ANOMALY_FEATURE] = models.iforest_score(art.iforest, X)
    feats[config.ANOMALY_FEATURE] = X[config.ANOMALY_FEATURE].to_numpy()
    feats["fraud_score"] = models.lgbm_score(art.lgbm, X)
    feats["risk_level"] = models.risk_level(feats["fraud_score"].to_numpy(), art.thresholds)
    feats["flagged"] = feats["fraud_score"] >= art.thresholds["flag"]

    if with_reasons and len(feats):
        expl = explain.explainer(art.lgbm)
        sv = explain.shap_values(expl, X[config.MODEL_FEATURES])
        top = np.argsort(-sv, axis=1)[:, :3]
        cols = np.array(config.MODEL_FEATURES)
        feats["top_reasons"] = [
            ", ".join(explain.FEATURE_LABELS.get(c, c) for c, v in zip(cols[r], sv[i, r]) if v > 0)
            for i, r in enumerate(top)
        ]
    return feats
