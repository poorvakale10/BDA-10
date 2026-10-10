# Credit Card Fraud Detection Using Big Data Analytics

**BDA Assignment No. 10 — Technical Paper Project**

## Authors
- Poorva Kale
- Shreya Sathish

## Problem
Detect rare fraudulent transactions in a highly imbalanced transaction dataset. The prototype compares Logistic Regression and Random Forest using precision, recall, F1-score, confusion matrix, and average precision.

## Dataset
Use the public [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Download `creditcard.csv` and place it in `data/`. The dataset is not redistributed in this repository. Follow the dataset's terms and access requirements.

Expected target column: `Class` (`1` = fraud, `0` = legitimate).

## Run locally
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Put `creditcard.csv` in `data/`.
5. Run:
   ```bash
   python src/fraud_detection.py
   ```

The script prints the classification report, confusion matrix, and average precision for each model. Run it before adding numerical results to the paper; do not invent or copy unverified scores.

## Method
- Remove exact duplicate rows and rows without a target.
- Use a stratified 80/20 train-test split.
- Scale features for Logistic Regression.
- Use class weights to account for imbalance.
- Evaluate both models on the same held-out test set.

## Limitations
This is an educational prototype, not a production banking system. The data are historical and anonymized. A random split does not model future deployment as well as time-based validation. A real system needs security controls, drift monitoring, human review, and threshold selection based on business costs.

## Repository structure
```text
.
├── data/                 # Place downloaded CSV here; do not commit private data
├── src/fraud_detection.py
├── requirements.txt
└── README.md
```

## Contributions
Update this section to reflect the actual work:
- Poorva Kale: problem formulation, literature review, methodology, report preparation.
- [Partner Name]: implementation testing, evaluation analysis, paper review.

## Ethical and security note
Do not upload real customer data, credentials, API keys, or personally identifiable information. This code should not be used to make consequential decisions without independent validation.
