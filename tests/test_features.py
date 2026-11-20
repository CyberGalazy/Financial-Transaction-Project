"""Behavioural features must only use earlier transactions of the same card."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud import features  # noqa: E402


def _toy():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "date": pd.to_datetime(["2015-01-01 10:00", "2015-01-01 10:30", "2015-01-02 10:15",
                                "2015-01-01 10:10", "2015-01-03 12:00"]),
        "client_id": [7, 7, 7, 7, 8],
        "card_id": [100, 100, 100, 101, 200],
        "amount": np.array([10, 20, 1000, 50, 5], dtype="float32"),
        "use_chip": pd.Categorical(["Swipe Transaction"] * 5),
        "merchant_id": [1, 1, 2, 3, 4],
        "merchant_city": pd.Categorical(["A", "A", "ONLINE", "B", "C"]),
        "merchant_state": pd.Categorical(["CA", "CA", None, "NY", "Italy"]),
        "zip": np.nan,
        "mcc": np.array([5411, 5411, 5732, 5812, 5812], dtype="int16"),
        "errors": pd.Categorical([None, "Bad PIN", None, None, "Bad CVV,Insufficient Balance"]),
    })
    return df


def test_behaviour_is_past_only():
    df = features.add_transaction_features(_toy(), {})
    out = features.add_behavioral_features(df).set_index("id")
    # card 100: txns 1 (10), 2 (20), 3 (1000)
    assert out.loc[1, "card_txn_number"] == 0
    assert np.isnan(out.loc[1, "card_prev_mean_amount"])
    assert out.loc[2, "card_prev_mean_amount"] == 10
    assert out.loc[3, "card_prev_mean_amount"] == 15          # mean of 10 and 20, not including 1000
    assert out.loc[2, "txn_count_1h"] == 1 and out.loc[1, "txn_count_1h"] == 0
    assert out.loc[3, "txn_count_24h"] == 1                    # txn 1 is 24.25h earlier, txn 2 is 23.75h earlier
    assert out.loc[2, "amount_sum_24h"] == 10
    assert out.loc[2, "new_merchant_for_card"] == 0 and out.loc[3, "new_merchant_for_card"] == 1
    assert out.loc[2, "card_prev_error_count"] == 0 and out.loc[3, "card_prev_error_count"] == 1
    # client 7 uses two cards: txn 2 at 10:30 sees 10:00 (card 100) and 10:10 (card 101)
    assert out.loc[2, "client_txn_count_24h"] == 2
    assert out.loc[4, "txn_count_1h"] == 0                     # other card's history not counted
    assert out.loc[5, "is_foreign"] == 1 and out.loc[3, "is_online"] == 1
    assert out.loc[5, "err_bad_cvv"] == 1 and out.loc[5, "n_errors"] == 2
