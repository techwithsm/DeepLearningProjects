# Lending Club Loan Default Prediction

Predicts whether a Lending Club loan will **default** (not be fully paid back) with a deep learning
model built in **Keras / TensorFlow**. It is trained on historical loan data from 2007–2015.

> Course-end project, *Deep Learning with Keras and TensorFlow*. **Domain:** Finance.

---

## Contents

- [Problem statement](#problem-statement)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Logging](#logging)
- [Testing](#testing)
- [Design decisions](#design-decisions)
- [Limitations and future work](#limitations-and-future-work)

---

## Problem statement

Lending platforms such as Lending Club need to predict which loans will default. Each default is a
direct loss for investors, and rejecting a creditworthy borrower is lost business. The task is to
build a deep learning model that estimates the chance of default for future loans.

The main challenge is **class imbalance**: only 16% of loans default. A model that always predicts
"fully paid" scores 84% accuracy but catches no defaults, so plain accuracy is the wrong yardstick.

The assignment consists of four tasks:

| Task | Description | Where in the code |
|------|-------------|-------------------|
| 1. Feature transformation | Convert categorical values to discrete numbers | `features.one_hot_encode` |
| 2. Exploratory data analysis | Analyse how each factor relates to default | `eda.py` |
| 3. Additional feature engineering | Find strongly correlated features and drop them | `features.find_correlated_features` |
| 4. Modeling | Deep learning model with Keras / TensorFlow | `modeling.py` |

## Dataset

`DataSet/loan_data.csv` contains **9,578 loans**, 13 features and 1 target. It has no missing
values and no duplicate rows.

| Column | Description |
|--------|-------------|
| `credit.policy` | 1 if the borrower meets Lending Club's underwriting criteria, else 0 |
| `purpose` | Loan purpose: `credit_card`, `debt_consolidation`, `educational`, `major_purchase`, `small_business`, `home_improvement`, `all_other` |
| `int.rate` | Interest rate as a proportion (11% → 0.11) |
| `installment` | Monthly installment owed |
| `log.annual.inc` | Natural log of self-reported annual income |
| `dti` | Debt-to-income ratio |
| `fico` | FICO credit score |
| `days.with.cr.line` | Days the borrower has had a credit line |
| `revol.bal` | Revolving balance (unpaid at end of credit-card billing cycle) |
| `revol.util` | Revolving line utilisation rate |
| `inq.last.6mths` | Creditor inquiries in the last 6 months |
| `delinq.2yrs` | Times 30+ days past due in the past 2 years |
| `pub.rec` | Derogatory public records (bankruptcies, tax liens, judgments) |
| **`not.fully.paid`** | **Target:** 1 = default, 0 = fully paid (16.0% / 84.0%) |

`DataSet/input.csv` and `DataSet/output.csv` are pre-encoded copies of the same data that came with
the assignment: features with `purpose` one-hot encoded, and a one-hot encoded target. They have no
header row, and the code does not use them.

> **Note:** `loan_data.csv` uses old Mac-style (`\r`) line endings. pandas reads it correctly, but
> `wc -l` reports 0 lines.

## Approach

```
loan_data.csv
   │
   ▼
validate ──► stratified split 70 / 15 / 15 (train / validation / test)
                 │
                 ▼
     FeaturePipeline.fit(train)               ◄── learned on the training split only
       • one-hot encode `purpose` (Task 1)
       • drop one feature of each pair with |r| > 0.7 (Task 3) → drops `fico`
       • log1p(`revol.bal`) to tame skew
       • median imputation + StandardScaler
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  Logistic regression   Keras MLP (Task 4)
  (baseline)            64 → 32 → 16, ReLU, BatchNorm, Dropout 0.3, L2
                        class weights · early stopping on val AUC · ReduceLROnPlateau
                           │
                           ▼
              decision threshold tuned for F1 on validation set
                           │
                           ▼
              final evaluation on untouched test set
```

**Key EDA findings**
- Default rate is highest for **small-business** loans (27.8%) and **educational** loans (20.1%),
  and lowest for credit-card and major-purchase loans (about 11%).
- Borrowers who don't meet the **credit policy** default at 27.8%, against 13.2% for those who do.
- Default risk rises with **interest rate** and **recent credit inquiries** (11.7% with no
  inquiries, 35.1% with six or more), and falls with **FICO score**.
- `int.rate` and `fico` are strongly correlated (r = −0.71), because Lending Club prices loans from
  the credit score. `fico` is dropped because it is the weaker predictor of the target.

## Results

Evaluated on a held-out test set of 1,437 loans, 230 of which defaulted:

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|-----------|---------|--------|-----------|--------|----|
| Logistic regression (baseline) | 0.50 | 0.657 | 0.310 | 0.241 | 0.526 | 0.330 |
| **Keras neural network** | 0.47 | 0.645 | 0.279 | 0.236 | **0.570** | **0.333** |

Precision, recall and F1 are for the default class. Random-guess PR-AUC is 0.16.

- The network catches **57% of defaults (131 of 230)**. The naive "always fully paid" model
  catches none.
- About 1 in 4 flagged loans actually defaults. The decision threshold controls this trade-off
  between catching defaults and flagging good loans.
- The network performs **on par with logistic regression**. The 13 available features carry
  mostly linear signal, and ROC-AUC around 0.65 is typical for this dataset.

Running the pipeline writes all figures to `reports/figures/` and the metrics to
`reports/metrics.json`. They include EDA plots, training curves, ROC/PR curves and the confusion
matrix.

## Project structure

```
LendingClubLoanDefault/
├── DataSet/
│   ├── loan_data.csv                        source data used by the code
│   ├── input.csv, output.csv                pre-encoded copies (not used)
├── Lending_Club_Loan_Default_Prediction.ipynb   step-by-step walkthrough of the 4 tasks
├── src/loan_default/
│   ├── config.py          all paths, preprocessing settings and hyperparameters (dataclasses)
│   ├── logging_utils.py   logging setup (console + file) and timed `log_step` context manager
│   ├── data.py            load_data, validate_data, split_data
│   ├── features.py        one_hot_encode, find_correlated_features, FeaturePipeline
│   ├── eda.py             EDA plot functions (each returns a matplotlib Figure)
│   ├── modeling.py        build_model, train_model, class weights, baseline, EpochLogger callback
│   ├── evaluation.py      metrics, threshold search, evaluation plots
│   ├── predictor.py       LoanDefaultPredictor: save / load / predict
│   ├── pipeline.py        run_training and run_eda (end-to-end orchestration)
│   ├── cli.py             command-line interface (train / eda / predict)
│   └── __main__.py        enables `python -m loan_default`
├── tests/                 pytest unit tests (synthetic data, no dataset needed)
├── pyproject.toml         package metadata, dependencies, pytest config
└── requirements.txt
```

Generated at run time and git-ignored:

| Folder | Contents |
|--------|----------|
| `models/` | `model.keras`, `features.joblib` (fitted feature pipeline), `metadata.json` (threshold, features, metrics) |
| `reports/` | `metrics.json`, `figures/eda/*.png`, `figures/model/*.png` |
| `logs/` | One timestamped log file per run |

## Setup

Requires **Python 3.10–3.13**. TensorFlow does not support 3.14 yet.

```bash
cd LendingClubLoanDefault
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev]"           # or: pip install -r requirements.txt
```

## Usage

### Notebook

Open `Lending_Club_Loan_Default_Prediction.ipynb` and select the virtual environment as the
kernel. The notebook walks through each task and calls the functions in `src/loan_default/`, so
there is no duplicated logic.

### Command line

```bash
# Full pipeline: train, evaluate, save model + metrics + figures (add --eda for EDA figures)
python -m loan_default train --eda

# EDA figures only
python -m loan_default eda

# Score new loans (CSV with the 13 feature columns; the target column is not needed)
python -m loan_default predict --input new_loans.csv --output scored.csv

# Options
python -m loan_default --log-level DEBUG train --epochs 50 --no-plots
python -m loan_default --help
```

`predict` adds two columns: `default_probability` and `predicted_default`, which is 1 when the
probability is at or above the tuned threshold.

### From Python

```python
from loan_default.predictor import LoanDefaultPredictor
import pandas as pd

predictor = LoanDefaultPredictor.load("models")
predictor.predict(pd.read_csv("new_loans.csv"))
```

### Changing settings

All tunable values are in `src/loan_default/config.py`: the data path, split sizes, correlation
threshold, network layers, learning rate, epochs, early-stopping patience and the threshold search
grid.

## Logging

Every module logs through the shared `loan_default` logger, configured in `logging_utils.py`:

- **Console:** INFO level by default; change it with `--log-level`.
- **File:** `logs/<command>_<timestamp>.log` always records DEBUG detail. This includes every
  training epoch's metrics, the model summary and the full configuration.
- Each pipeline step logs when it starts, how long it took, and the full traceback if it fails.
- Data-quality issues are logged as warnings: missing values, duplicates, unseen categories at
  prediction time, negative balances.

Sample output:

```
2026-10-31 14:34:29 | INFO | loan_default.features | int.rate vs fico: r=-0.709 -> drop 'fico' (|r with target| 0.146 < 0.163)
2026-10-31 14:34:39 | INFO | loan_default.modeling | Stopped after 40 epochs; restored best epoch 15 (val_auc=0.6560, val_loss=0.6307)
2026-10-31 14:34:40 | INFO | loan_default.evaluation | Neural Network @ 0.47 | ROC-AUC 0.6452 | PR-AUC 0.2792 | precision 0.236 | recall 0.570 | F1 0.333
```

## Testing

```bash
pytest
```

The 18 tests run on a small synthetic dataset, so they don't need `loan_data.csv`. They cover data
validation, stratified splitting, encoding (including unseen categories), correlation-based
feature dropping, the feature pipeline (column order, missing values, unfitted use), metrics,
threshold selection, and a model train → save → load → predict round trip.

## Design decisions

| Decision | Reason |
|----------|--------|
| One-hot encoding rather than label encoding for `purpose` | `purpose` is nominal. Integer codes would imply an order the network would treat as meaningful. |
| Preprocessing fitted on the training split only | Prevents test-set information leaking into scaling or feature selection |
| Class weights rather than oversampling | Counters the 5 : 1 imbalance without adding synthetic rows or distorting validation data |
| Model selection on validation AUC, not accuracy | Accuracy is misleading under imbalance |
| Threshold tuned on the validation set | Class weights shift probabilities, so 0.5 isn't optimal. Tuning on test would leak information. |
| Logistic regression baseline | Shows whether the deep model adds value over a linear model |
| Fixed seeds + deterministic TF ops | Re-running gives the same results |

## Limitations and future work

- Only 13 borrower features, so the predictive signal is limited.
- With 230 test defaults, metric differences of about ±0.01 are within noise. k-fold
  cross-validation would give a more reliable comparison.
- Possible next steps:
  - Hyperparameter search with Keras Tuner.
  - Focal loss or SMOTE as alternatives to class weights.
  - Gradient-boosted trees (XGBoost / LightGBM) for comparison.
  - A cost-sensitive threshold based on the real cost of a default versus a rejected good loan.
