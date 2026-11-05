import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_b"
INPUT_PATH = INPUT_DIR / "customer_features.csv"
OUTPUT_PATH = INPUT_DIR / "customer_segments.csv"
METRICS_PATH = INPUT_DIR / "segmentation_metrics.json"
MODEL_DIR = PROJECT_ROOT / "models" / "part_b"
MODEL_PATH = MODEL_DIR / "kmeans_segmentation.joblib"

ID_COLUMNS = {
    "customer_id",
    "customerid",
    "customer",
    "id",
}
EXCLUDED_COLUMNS = ID_COLUMNS | {
    "churn",
    "churned",
    "churn_label",
    "is_churned",
    "target",
    "label",
    "segment",
    "cluster",
}


def load_features():
    """Load customer-level features and verify the customer identifier."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Create it by running feature_engineering.py first."
        )

    dataframe = pd.read_csv(INPUT_PATH)

    if dataframe.empty:
        raise ValueError("The customer feature file contains no rows.")

    if "customer_id" not in dataframe.columns:
        raise ValueError(
            "customer_features.csv must contain a 'customer_id' column."
        )

    if dataframe["customer_id"].isna().any():
        raise ValueError("customer_id contains missing values.")

    if dataframe["customer_id"].duplicated().any():
        raise ValueError(
            "Expected one row per customer, but duplicate customer_id values "
            "were found. Fix feature engineering before clustering."
        )

    return dataframe


def select_numeric_features(dataframe):
    """Select numeric behavioural features and exclude IDs/target labels."""
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    feature_columns = [
        column
        for column in numeric_columns
        if column.strip().lower() not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No numeric customer-behaviour features were found. "
            "Check feature_engineering.py output."
        )

    features = dataframe[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain no usable values or have no variation.
    usable_columns = [
        column
        for column in features.columns
        if features[column].notna().any()
        and features[column].nunique(dropna=True) > 1
    ]
    features = features[usable_columns]

    if features.shape[1] == 0:
        raise ValueError(
            "No varying numeric features are available for clustering."
        )

    return features, usable_columns


def choose_cluster_count(scaled_features):
    """Choose K using silhouette score across feasible candidate values."""
    row_count = scaled_features.shape[0]
    if row_count < 3:
        raise ValueError(
            "At least 3 customers are required for silhouette-based clustering."
        )

    max_k = min(6, row_count - 1)
    candidate_results = []

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )
        labels = model.fit_predict(scaled_features)

        # Silhouette is undefined if the model creates only one cluster
        # or one cluster per observation.
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(unique_labels) >= row_count:
            continue

        score = float(silhouette_score(scaled_features, labels))
        candidate_results.append({
            "n_clusters": int(k),
            "silhouette_score": score,
            "inertia": float(model.inertia_),
        })

    if not candidate_results:
        raise ValueError(
            "Could not find a valid KMeans clustering. "
            "Check the number and variation of customer features."
        )

    best_result = max(
        candidate_results,
        key=lambda result: result["silhouette_score"],
    )
    return best_result, candidate_results


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART B: CUSTOMER SEGMENTATION")
    print("=" * 65)

    dataframe = load_features()
    features, feature_columns = select_numeric_features(dataframe)

    preprocessing = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    scaled_features = preprocessing.fit_transform(features)

    best_result, candidate_results = choose_cluster_count(scaled_features)
    selected_k = best_result["n_clusters"]

    kmeans = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(scaled_features)

    result = dataframe[["customer_id"]].copy()
    result["cluster"] = cluster_labels

    # Add selected features to the output to make segment profiling possible.
    for column in feature_columns:
        result[column] = features[column].to_numpy()

    segment_profile = (
        result.groupby("cluster")[feature_columns]
        .mean(numeric_only=True)
        .round(3)
    )
    segment_sizes = (
        result["cluster"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)
    segment_profile.to_csv(INPUT_DIR / "segment_profile.csv")

    model_artifact = {
        "preprocessing": preprocessing,
        "kmeans": kmeans,
        "feature_columns": feature_columns,
        "selected_k": selected_k,
        "random_state": 42,
    }
    joblib.dump(model_artifact, MODEL_PATH)

    metrics = {
        "stage": "Part B - Customer segmentation",
        "input_file": str(INPUT_PATH.relative_to(PROJECT_ROOT)),
        "customer_count": int(len(dataframe)),
        "feature_columns": feature_columns,
        "selected_cluster_count": int(selected_k),
        "selected_silhouette_score": best_result["silhouette_score"],
        "selected_inertia": best_result["inertia"],
        "candidate_results": candidate_results,
        "cluster_sizes": {
            str(cluster): int(count)
            for cluster, count in segment_sizes.items()
        },
        "outputs": [
            str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            str((INPUT_DIR / "segment_profile.csv").relative_to(PROJECT_ROOT)),
            str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        ],
        "notes": [
            "Cluster labels are arbitrary identifiers, not ranked customer values.",
            "Numeric target/label columns and customer identifiers are excluded.",
            "Default K search range is 2-6; align it with assignment checkpoints if specified.",
        ],
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print(f"Customers clustered: {len(dataframe)}")
    print(f"Features used:       {', '.join(feature_columns)}")
    print(f"Selected clusters:   {selected_k}")
    print(f"Silhouette score:    {best_result['silhouette_score']:.4f}")
    print(f"Cluster sizes:       {metrics['cluster_sizes']}")
    print(f"Assignments saved:   {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Profile saved:       {(INPUT_DIR / 'segment_profile.csv').relative_to(PROJECT_ROOT)}")
    print(f"Metrics saved:       {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Model saved:         {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 65)
    print("CUSTOMER SEGMENTATION COMPLETED")


if __name__ == "__main__":
    main()
