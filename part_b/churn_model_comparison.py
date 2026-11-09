import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "part_b"
MODEL_DIR = ROOT / "models" / "part_b"

TARGET = "churn"
NON_FEATURE_COLUMNS = {"customer_id", "snapshot_date", TARGET}


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def main():
    input_path = OUTPUT_DIR / "customer_churn_training.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            "Customer churn data not found. Run "
            "'python part_b/churn_target.py' first."
        )

    data = pd.read_csv(input_path)
    required = {"customer_id", "snapshot_date", TARGET}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Churn dataset is missing columns: {sorted(missing)}")

    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=["snapshot_date", TARGET])
    data[TARGET] = data[TARGET].astype(int)

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 and 1.")

    snapshot_dates = sorted(data["snapshot_date"].dropna().unique())
    if len(snapshot_dates) < 2:
        raise ValueError(
            "At least two snapshot dates are required for a time-based split."
        )

    test_date = snapshot_dates[-1]
    train_data = data[data["snapshot_date"] < test_date].copy()
    test_data = data[data["snapshot_date"] == test_date].copy()

    feature_columns = [
        column
        for column in data.select_dtypes(include=[np.number]).columns
        if column not in NON_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available.")

    X_train = train_data[feature_columns]
    y_train = train_data[TARGET]
    X_test = test_data[feature_columns]
    y_test = test_data[TARGET]

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "Both train and test sets must contain churn and non-churn examples."
        )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {}
    prediction_rows = []

    for model_name, model in models.items():
        result = evaluate_model(model, X_train, y_train, X_test, y_test)
        metrics[model_name] = {
            key: value
            for key, value in result.items()
            if key not in {"predictions", "probabilities"}
        }

        for row_index, (_, row) in enumerate(test_data.iterrows()):
            prediction_rows.append(
                {
                    "model": model_name,
                    "customer_id": row["customer_id"],
                    "snapshot_date": row["snapshot_date"].strftime("%Y-%m-%d"),
                    "actual_churn": int(y_test.iloc[row_index]),
                    "predicted_churn": int(result["predictions"][row_index]),
                    "churn_probability": float(result["probabilities"][row_index]),
                }
            )

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "target": TARGET,
                "train_snapshot_dates": [
                    pd.Timestamp(date).strftime("%Y-%m-%d")
                    for date in snapshot_dates[:-1]
                ],
                "test_snapshot_date": pd.Timestamp(test_date).strftime("%Y-%m-%d"),
            },
            MODEL_DIR / f"{model_name}_churn.joblib",
        )

    predictions_path = OUTPUT_DIR / "churn_model_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    selected_model = max(
        metrics,
        key=lambda name: metrics[name]["pr_auc_average_precision"],
    )

    report = {
        "target_definition": "No order in the 30 days following each customer snapshot.",
        "train_snapshot_dates": [
            pd.Timestamp(date).strftime("%Y-%m-%d")
            for date in snapshot_dates[:-1]
        ],
        "test_snapshot_date": pd.Timestamp(test_date).strftime("%Y-%m-%d"),
        "train_rows": int(len(train_data)),
        "test_rows": int(len(test_data)),
        "feature_columns": feature_columns,
        "models": metrics,
        "selected_model_by_pr_auc": selected_model,
        "limitations": [
            "The target is derived from future order activity, not a supplied ground-truth churn label.",
            "The latest snapshot is an out-of-time holdout; customers may appear in earlier training snapshots.",
            "The short synthetic observation period limits generalisation to real customer behaviour.",
        ],
    }

    report_path = OUTPUT_DIR / "churn_model_comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Churn model comparison completed.")
    for model_name, values in metrics.items():
        print(
            f"{model_name}: ROC-AUC={values['roc_auc']:.4f}, "
            f"PR-AUC={values['pr_auc_average_precision']:.4f}, "
            f"F1={values['f1']:.4f}"
        )
    print(f"Selected model by PR-AUC: {selected_model}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
