import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "part_b"

CUSTOMERS_PATH = PROCESSED_DIR / "customers_clean.csv"
ORDERS_PATH = PROCESSED_DIR / "orders_clean.csv"
OUTPUT_PATH = OUTPUT_DIR / "customer_features.csv"
REPORT_PATH = OUTPUT_DIR / "feature_engineering_report.json"


def load_input_data():
    """Load cleaned Part A data and validate the required columns."""
    if not CUSTOMERS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CUSTOMERS_PATH}. Run Part A cleaning.py first."
        )
    if not ORDERS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ORDERS_PATH}. Run Part A cleaning.py first."
        )

    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)

    required_customer = {"customer_id"}
    required_order = {
        "order_id", "customer_id", "quantity", "order_ts", "status",
        "discount_pct", "order_amount", "delivery_minutes",
    }

    missing_customer = required_customer - set(customers.columns)
    missing_order = required_order - set(orders.columns)

    if missing_customer:
        raise ValueError(
            f"customers_clean.csv is missing columns: {sorted(missing_customer)}"
        )
    if missing_order:
        raise ValueError(
            f"orders_clean.csv is missing columns: {sorted(missing_order)}"
        )
    if customers["customer_id"].isna().any():
        raise ValueError("customers_clean.csv contains missing customer_id values.")

    customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
    orders["customer_id"] = orders["customer_id"].astype(str).str.strip()

    if customers["customer_id"].duplicated().any():
        raise ValueError("customers_clean.csv must contain one row per customer.")

    return customers, orders


def build_customer_features(customers, orders):
    """Aggregate order behaviour and combine it with customer attributes."""
    customers = customers.copy()
    orders = orders.copy()

    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    for column in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
    orders["status"] = orders["status"].astype(str).str.strip().str.lower()

    valid_timestamps = orders["order_ts"].dropna()
    if valid_timestamps.empty:
        raise ValueError("No valid order_ts values found in orders_clean.csv.")

    # Use the latest observed order date as a reproducible feature snapshot.
    snapshot_date = valid_timestamps.max().normalize()

    known_ids = set(customers["customer_id"])
    unknown_customer_orders = int((~orders["customer_id"].isin(known_ids)).sum())
    orders = orders[orders["customer_id"].isin(known_ids)].copy()

    grouped = orders.groupby("customer_id")
    aggregates = grouped.agg(
        total_orders=("order_id", "nunique"),
        completed_orders=("status", lambda x: int((x == "completed").sum())),
        cancelled_orders=("status", lambda x: int((x == "cancelled").sum())),
        pending_orders=("status", lambda x: int((x == "pending").sum())),
        total_quantity=("quantity", "sum"),
        average_quantity_per_order=("quantity", "mean"),
        total_order_amount=("order_amount", "sum"),
        average_order_amount=("order_amount", "mean"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
        maximum_delivery_minutes=("delivery_minutes", "max"),
        first_order_ts=("order_ts", "min"),
        last_order_ts=("order_ts", "max"),
    ).reset_index()

    aggregates["completed_order_rate"] = (
        aggregates["completed_orders"]
        / aggregates["total_orders"].replace(0, np.nan)
    )
    aggregates["cancelled_order_rate"] = (
        aggregates["cancelled_orders"]
        / aggregates["total_orders"].replace(0, np.nan)
    )
    aggregates["days_since_last_order"] = (
        snapshot_date - aggregates["last_order_ts"].dt.normalize()
    ).dt.days
    aggregates["customer_order_span_days"] = (
        aggregates["last_order_ts"].dt.normalize()
        - aggregates["first_order_ts"].dt.normalize()
    ).dt.days

    features = customers.merge(
        aggregates, on="customer_id", how="left", validate="one_to_one"
    )

    # Customers with no orders remain in the output.
    for column in [
        "total_orders", "completed_orders", "cancelled_orders",
        "pending_orders", "total_quantity", "total_order_amount",
    ]:
        features[column] = features[column].fillna(0)

    for column in ["completed_order_rate", "cancelled_order_rate"]:
        features[column] = features[column].fillna(0)

    if "age" in features.columns:
        features["age"] = pd.to_numeric(features["age"], errors="coerce")

    if "signup_date" in features.columns:
        signup_dates = pd.to_datetime(features["signup_date"], errors="coerce")
        features["customer_tenure_days"] = (
            snapshot_date - signup_dates.dt.normalize()
        ).dt.days
        features.loc[features["customer_tenure_days"] < 0, "customer_tenure_days"] = np.nan

    features.drop(
        columns=["first_order_ts", "last_order_ts"],
        errors="ignore",
        inplace=True,
    )

    numeric_columns = features.select_dtypes(include=[np.number]).columns
    features[numeric_columns] = features[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    report = {
        "stage": "Part B - Customer feature engineering",
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "customer_count": int(len(features)),
        "order_rows_used": int(len(orders)),
        "customers_without_orders": int(features["total_orders"].eq(0).sum()),
        "orders_with_unknown_customer_id_excluded": unknown_customer_orders,
        "feature_columns": [c for c in features.columns if c != "customer_id"],
        "outputs": [
            str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        ],
        "notes": [
            "One output row is retained for every customer in customers_clean.csv.",
            "Order counts use distinct order_id values; status counts use cleaned order rows.",
            "The snapshot date is the latest valid order timestamp in orders_clean.csv.",
            "No churn target is created; apply the assignment's exact churn definition separately.",
            "Categorical customer attributes are retained for interpretation; numeric-only models ignore them.",
        ],
    }
    return features, report


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART B: FEATURE ENGINEERING")
    print("=" * 65)

    customers, orders = load_input_data()
    features, report = build_customer_features(customers, orders)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print(f"Customers processed: {report['customer_count']}")
    print(f"Order rows used: {report['order_rows_used']}")
    print(f"Snapshot date: {report['snapshot_date']}")
    print(f"Features created: {len(report['feature_columns'])}")
    print(f"Customers without orders: {report['customers_without_orders']}")
    print(f"Feature file: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Report file: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("FEATURE ENGINEERING COMPLETED")


if __name__ == "__main__":
    main()
