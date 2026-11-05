"""
Part B - Customer churn prediction.

Run from the repository root:
    python part_b/churn_prediction.py

Input:
    data/processed/part_b/customer_features.csv

Required input columns:
    customer_id
    churn

The 'churn' column must be a ground-truth target label created according to
the exact churn definition in the assignment. This script deliberately does
not invent a churn definition from order inactivity.

Outputs:
    data/processed/part_b/churn_predictions.csv
    data/processed/part_b/churn_model_metrics.json
    models/part_b/churn_prediction.joblib

Only numeric customer-behaviour features are used by this baseline model.
Customer IDs and target/label columns are excluded from model inputs.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_b"
INPUT_PATH = INPUT_DIR / "customer_features.csv"
PREDICTIONS_PATH = INPUT_DIR / "churn_predictions.csv"
METRICS_PATH = INPUT_DIR / "churn_model_metrics.json"
MODEL_DIR = PROJECT_ROOT / "models" / "part_b"
MODEL_PATH = MODEL_DIR / "churn_prediction.joblib"

TARGET_COLUMN = "churn"
ID_COLUMNS = {"customer_id", "customerid", "customer", "id"}
TARGET_ALIASES = {
    "churn", "churned", "churn_label", "is_churned",
    "target", "label",
}
RANDOM_STATE = 42


def load_data():
    """Load customer features and check the required churn target."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run feature_engineering.py first and ensure it creates "
            "customer_features.csv with a churn target."
        )

    dataframe = pd.read_csv(INPUT_PATH)

    if dataframe.empty:
        raise ValueError("customer_features.csv contains no rows.")

    if "customer_id" not in dataframe.columns:
        raise ValueError("Input must contain a 'customer_id' column.")

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            "Input must contain a 'churn' target column. Create this label "
            "using the assignment's exact churn definition; this script "
            "will not infer or invent the definition."
        )

    if dataframe["customer_id"].isna().any():
        raise ValueError("customer_id contains missing values.")

    if dataframe["customer_id"].duplicated().any():
        raise ValueError(
            "Expected one row per customer, but duplicate customer IDs "
            "were found."
        )

    dataframe = dataframe.dropna(subset=[TARGET_COLUMN]).copy()
    if dataframe.empty:
        raise ValueError("No rows have a non-missing churn target.")

    return dataframe


def normalize_target(target):
    """Convert common binary target representations to integer 0/1."""
    if pd.api.types.is_numeric_dtype(target):
        values = set(pd.to_numeric(target, errors="coerce").dropna().unique())
        if values.issubset({0, 1}):
            return target.astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "not_churned": 0,
        "churned": 1,
        "active": 0,
        "inactive": 1,
    }
    normalized = target.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        unexpected = sorted(
            target[normalized.isna()].astype(str).unique().tolist()
        )
        raise ValueError(
            "The churn target must be binary (0/1 or a recognized "
            f"yes/no label). Unexpected values: {unexpected}"
        )

    return normalized.astype(int)


def select_features(dataframe):
    """Select numeric predictors, excluding IDs and target-like columns."""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    excluded = ID_COLUMNS | TARGET_ALIASES

    feature_columns = [
        column
        for column in numeric_columns
        if column.strip().lower() not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric customer-behaviour features were found. "
            "Check feature_engineering.py."
        )

    features = dataframe[feature_columns].replace([np.inf, -np.inf], np.nan)

    usable_columns = [
        column
        for column in features.columns
        if features[column].notna().any()
        and features[column].nunique(dropna=True) > 1
    ]
    features = features[usable_columns]

    if features.shape[1] == 0:
        raise ValueError("No usable varying numeric features remain.")

    return features, usable_columns


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART B: CHURN PREDICTION")
    print("=" * 65)

    dataframe = load_data()
    target = normalize_target(dataframe[TARGET_COLUMN])

    class_counts = target.value_counts().to_dict()
    if len(class_counts) != 2:
        raise ValueError(
            "Churn prediction requires both classes (0 = not churned, "
            "1 = churned) in the labeled dataset."
        )
    if min(class_counts.values()) < 2:
        raise ValueError(
            "Each churn class needs at least two examples for a "
            "stratified train/test split."
        )

    features, feature_columns = select_features(dataframe)

    # Keep the test set stratified so both churn classes are represented.
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        features,
        target,
        dataframe["customer_id"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        )),
    ])

    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics = {
        "stage": "Part B - Churn prediction",
        "target_column": TARGET_COLUMN,
        "target_definition_note": (
            "The churn label must be generated using the assignment's "
            "specified definition. This script does not create the label."
        ),
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "customer_count_with_labels": int(len(dataframe)),
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "feature_columns": feature_columns,
        "class_distribution": {
            str(label): int(count)
            for label, count in class_counts.items()
        },
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(
            y_test, predicted, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predicted, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predicted, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_test, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predicted,
            labels=[0, 1],
            target_names=["not_churned", "churned"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
    else:
        metrics["roc_auc"] = None

    test_results = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "actual_churn": y_test.to_numpy(),
        "predicted_churn": predicted,
        "churn_probability": probabilities,
    }).sort_values("customer_id")

    # Refit on all labeled data after evaluating on the held-out test set.
    final_model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        )),
    ])
    final_model.fit(features, target)

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    test_results.to_csv(PREDICTIONS_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )
    joblib.dump({
        "model": final_model,
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
    }, MODEL_PATH)

    print(f"Customers with labels: {len(dataframe)}")
    print(f"Features used:         {', '.join(feature_columns)}")
    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Precision:             {metrics['precision']:.4f}")
    print(f"Recall:                {metrics['recall']:.4f}")
    print(f"F1 score:              {metrics['f1_score']:.4f}")
    print(f"ROC AUC:               {metrics['roc_auc']}")
    print(f"Test predictions:      {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metrics:               {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Model:                 {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 65)
    print("CHURN PREDICTION COMPLETED")


if __name__ == "__main__":
    main()
