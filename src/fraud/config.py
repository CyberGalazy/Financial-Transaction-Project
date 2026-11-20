"""Paths, split dates and feature lists shared by the pipeline, notebook and app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

KAGGLE_DATASET = "computingvictor/transactions-fraud-datasets"

TRANSACTIONS_CSV = RAW_DIR / "transactions_data.csv"
CARDS_CSV = RAW_DIR / "cards_data.csv"
USERS_CSV = RAW_DIR / "users_data.csv"
MCC_JSON = RAW_DIR / "mcc_codes.json"
LABELS_JSON = RAW_DIR / "train_fraud_labels.json"

CLEAN_TXN_PARQUET = PROCESSED_DIR / "transactions_clean.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features_labeled.parquet"
SCORED_TEST_PARQUET = PROCESSED_DIR / "scored_test.parquet"

# Time-based split: train on the past, tune on the next half-year, test on the most recent year.
# Fraud patterns drift strongly over time in this data (2012-16: mostly online fraud,
# 2017-19: mostly foreign-merchant fraud), so training includes H1 2018 to see the recent regime.
TRAIN_END = "2018-07-01"   # train:      date <  2018-07-01
VALID_END = "2019-01-01"   # validation: 2018-07 .. 2018-12
                           # test:       date >= 2019-01-01 (never used for fitting or tuning)

RANDOM_STATE = 42

ERROR_TYPES = [
    "Bad CVV", "Bad Card Number", "Bad Expiration", "Bad PIN",
    "Bad Zipcode", "Insufficient Balance", "Technical Glitch",
]

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC", "AA", "AE", "AP",
}

CATEGORICAL_FEATURES = ["use_chip", "mcc", "card_brand", "card_type", "gender"]

NUMERIC_FEATURES = [
    # transaction
    "amount", "log_abs_amount", "is_refund", "hour", "day_of_week", "is_night",
    "is_online", "is_foreign", "has_error", "n_errors",
    *[f"err_{e.lower().replace(' ', '_')}" for e in ERROR_TYPES],
    # behavioural (strictly past-only, per card)
    "secs_since_prev_txn", "txn_count_1h", "txn_count_24h", "amount_sum_24h",
    "card_txn_number", "card_prev_mean_amount", "amount_zscore_card", "amount_ratio_card_mean",
    "new_merchant_for_card", "new_mcc_for_card", "new_state_for_card", "card_prev_error_count",
    "client_txn_count_24h", "card_foreign_count_24h", "card_foreign_count_7d",
    "card_online_count_24h", "card_new_merchant_count_24h",
    # card
    "card_on_dark_web", "has_chip", "credit_limit", "amount_to_limit", "num_cards_issued",
    # user
    "credit_score", "yearly_income", "total_debt", "debt_to_income",
    "num_credit_cards", "income_to_area",
]

# Computed but NOT used by the models: they grow steadily with calendar time, so trees use them to
# memorise which era a transaction is from instead of learning fraud behaviour (hurts under drift).
TIME_PROXY_FEATURES = ["card_age_years", "years_since_pin_change", "age_at_txn", "card_txn_number", "card_expired"]
NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES if f not in TIME_PROXY_FEATURES]

ANOMALY_FEATURE = "if_anomaly_score"

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [ANOMALY_FEATURE]

# Columns shown in the app next to scores.
DISPLAY_COLUMNS = [
    "id", "date", "client_id", "card_id", "amount", "use_chip", "merchant_id",
    "merchant_city", "merchant_state", "mcc", "mcc_description", "errors",
]
