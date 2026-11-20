# ML Fraud Detection for Digital Financial Transactions

**Problem.** Digital financial transactions are increasing, and fraudulent activities can cause significant
financial losses. Traditional rule-based detection systems may fail to identify complex suspicious patterns.
This system uses machine learning to analyse transaction behaviour, identify anomalies and flag potentially
fraudulent activities.

**Data.** Kaggle [`computingvictor/transactions-fraud-datasets`](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets),
the dataset behind the [Financial transactions](https://www.kaggle.com/code/jigneshkirtinagda/financial-transactions)
EDA notebook: 13.3M card transactions (2010–2019), 2,000 users, 6,146 cards, ~8.9M fraud labels (≈0.15% fraud).

## How it works
1. **Clean & merge** users, cards, transactions (`src/fraud/data.py`, same cleaning as the reference notebook).
2. **Features** (`src/fraud/features.py`): transaction attributes, *past-only* behavioural features per card/client
   (velocity, spend deviation, first-time merchant/state, recent foreign/online activity), card and customer profile.
3. **Detectors** (`src/fraud/models.py`), compared on a time-based hold-out (train ≤ 2018-06, validate 2018 H2, test 2019):
   - Rule-based baseline (what traditional systems do)
   - Isolation Forest (unsupervised anomaly score; also fed to the supervised models)
   - Logistic Regression
   - **LightGBM** (main model)
4. **Thresholds** picked on validation: *High risk* = F1-optimal, *Medium* = ≥80% recall. SHAP explains each flag.

## Results (2019 hold-out, 777k transactions, 1,360 frauds)
| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision | Recall | False alarms |
|---|---|---|---|---|---|---|
| Rule-based baseline | 0.064 | 0.984 | 35% | 4.6% | 100% | 28,053 |
| Isolation Forest | 0.043 | 0.954 | 29% | 7.0% | 14% | 2,551 |
| Logistic Regression | 0.278 | 0.993 | 80% | 29% | 46% | 1,512 |
| **LightGBM** | **0.711** | **0.998** | **96%** | **78%** | 56% | **220** |

Precision/recall are at the High-risk threshold chosen on validation; the Medium (review) tier trades precision for more recall.

Notes:
* Fraud patterns **drift**: 2012–16 fraud is mostly online, 2017–19 mostly at foreign merchants. The rules look
  good in 2019 only because "foreign merchant" happens to match the current regime (and still raise 28k false alarms).
* Features that grow with calendar time (card age, years since PIN change, customer age) are computed but kept out of
  the models: trees used them to memorise the era instead of learning behaviour.
* Up-weighting the fraud class (`scale_pos_weight`) hurt ranking badly (validation AP 0.59 vs 0.84), so it is off.

## Setup & run
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py            # ~350 MB zip -> data/raw/ (public, no Kaggle token needed)
python scripts/train.py                    # full run, ~10 min; add --sample 500000 for a quick run
pytest -q                                  # leakage test for behavioural features
jupyter notebook notebooks/fraud_detection.ipynb
streamlit run app/streamlit_app.py
python scripts/build_report.py             # -> docs/Fraud_Detection_Project_Report.pdf
```

## Layout
```
src/fraud/      config, data loading, features, models, evaluation, SHAP explanations, scoring
scripts/        download_data.py, train.py
notebooks/      fraud_detection.ipynb — EDA, features, model comparison, explainability
docs/           Fraud_Detection_Project_Report.pdf (+ figures/)
app/            streamlit_app.py — overview, flagged list, per-transaction investigation, CSV scoring
models/         trained models + metrics (created by train.py)
data/           raw/ and processed/ (created by the scripts)
```
