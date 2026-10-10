"""
Credit Card Fraud Detection - Assignment 10
Run from repository root after placing creditcard.csv in data/.
Install: pip install -r requirements.txt
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score
)

DATA_PATH = Path("data/creditcard.csv")
TARGET = "Class"

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    print(f"\n{'=' * 18} {name} {'=' * 18}")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred))
    print(f"Average precision (PR-AUC summary): {average_precision_score(y_test, score):.4f}")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the Kaggle dataset and place creditcard.csv there."
        )
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found: {list(df.columns)}")
    df = df.drop_duplicates().dropna(subset=[TARGET])
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ))
    ])
    forest = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1, min_samples_leaf=2
    )

    for model in (logistic, forest):
        model.fit(X_train, y_train)

    evaluate("Logistic Regression", logistic, X_test, y_test)
    evaluate("Random Forest", forest, X_test, y_test)

if __name__ == "__main__":
    main()
