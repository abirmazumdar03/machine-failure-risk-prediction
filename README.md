# Predictive Maintenance for Industrial Machines

## Problem Statement

Industrial machines can fail under certain operating conditions. The goal of this project is to estimate whether a machine is likely to fail using current machine and process measurements, so maintenance teams can prioritize inspection before a breakdown occurs.

## Dataset

This project uses the UCI AI4I 2020 Predictive Maintenance Dataset.

Place the CSV file at:

```text
data/raw/ai4i2020.csv
```

If your filename is different, update `DEFAULT_DATA_PATH` in `src/data_processing.py`.

## Objective

Predict `Machine failure` using operating and product characteristics that would realistically be available before a failure.

## Approach

```text
Data Cleaning
-> EDA
-> Feature Engineering
-> Preprocessing
-> Classification
-> Evaluation
-> Explainability
```

## Models

- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation

Accuracy alone can be misleading because machine failures are usually rare. This project evaluates models using precision, recall, F1-score, ROC-AUC, and confusion matrices.

Recall for the failure class is especially important because a false negative means the model predicts normal operation even though the machine actually fails.

## Leakage Prevention

The main predictive setup excludes identifier columns:

- `UDI`
- `Product ID`

It also excludes failure-mode indicator columns:

- `TWF`
- `HDF`
- `PWF`
- `OSF`
- `RNF`

These failure-mode columns are closely tied to failure outcomes and would not be appropriate independent pre-failure inputs for the intended prediction scenario.

## Key Findings

Run `notebooks/predictive_maintenance.ipynb` after placing the dataset in `data/raw/ai4i2020.csv`. Populate this section only with results generated from the actual analysis.

## Limitations

- The dataset is synthetic.
- Model performance on this dataset does not guarantee real-world factory performance.
- Correlation and model importance do not establish causality.
- Real predictive maintenance deployment requires appropriate sensor, maintenance, and operational data from the target environment.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```text
notebooks/predictive_maintenance.ipynb
```
