import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "part_b"


def build_customer_snapshots():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = pd.read_csv(PROCESSED / "customers_clean.csv")
    orders = pd.read_csv(PROCESSED / "orders_clean.csv")
    products = pd.read_csv(PROCESSED / "products_clean.csv")

    customers["customer_id"] = customers["customer_id"].astype(str)
    orders["customer_id"] = orders["customer_id"].astype(str)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

    for column in ["order_amount", "quantity", "delivery_minutes", "discount_pct"]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")

    orders = orders.dropna(subset=["order_ts", "customer_id"])
    products = products[["product_id", "category"]].drop_duplicates("product_id")
    orders = orders.merge(products, on="product_id", how="left")

    first_date = orders["order_ts"].min().normalize()
    last_date = orders["order_ts"].max().normalize()

    # A full 30-day future window must be available for every snapshot.
    snapshots = pd.date_range(
        start=first_date + pd.Timedelta(days=60),
        end=last_date - pd.Timedelta(days=30),
        freq="ME",
    )

    if len(snapshots) == 0:
        raise ValueError(
            "Not enough order history to create a monthly snapshot with "
            "a complete 30-day future label window."
        )

    snapshot_frames = []

    for snapshot_date in snapshots:
        historical_orders = orders[orders["order_ts"] <= snapshot_date].copy()
        future_orders = orders[
            (orders["order_ts"] > snapshot_date)
            & (orders["order_ts"] <= snapshot_date + pd.Timedelta(days=30))
        ]

        if historical_orders.empty:
            continue

        grouped = historical_orders.groupby("customer_id")
        features = grouped.agg(
            total_orders=("order_id", "nunique"),
            total_spend=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
            total_quantity=("quantity", "sum"),
            cancelled_orders=("status", lambda values: int((values == "cancelled").sum())),
            completed_orders=("status", lambda values: int((values == "completed").sum())),
            average_delivery_minutes=("delivery_minutes", "mean"),
            average_discount_pct=("discount_pct", "mean"),
            first_order=("order_ts", "min"),
            last_order=("order_ts", "max"),
        ).reset_index()

        features["days_since_last_order"] = (
            snapshot_date - features["last_order"].dt.normalize()
        ).dt.days
        features["order_span_days"] = (
            features["last_order"] - features["first_order"]
        ).dt.days
        features["cancel_rate"] = (
            features["cancelled_orders"]
            / features["total_orders"].replace(0, np.nan)
        )

        # Mean gap between distinct orders for each customer.
        average_gaps = {}
        for customer_id, customer_orders in historical_orders.groupby("customer_id"):
            order_dates = (
                customer_orders.drop_duplicates("order_id")["order_ts"].sort_values()
            )
            average_gaps[customer_id] = (
                order_dates.diff().dt.total_seconds().div(86400).mean()
            )
        features["average_days_between_orders"] = (
            features["customer_id"].map(average_gaps)
        )

        category_counts = historical_orders.groupby("customer_id")["category"].nunique()
        features["distinct_product_categories"] = (
            features["customer_id"].map(category_counts)
        )

        snapshot = customers[["customer_id", "age"]].copy()
        snapshot["age"] = pd.to_numeric(snapshot["age"], errors="coerce")
        snapshot = snapshot.merge(features, on="customer_id", how="left")

        future_customer_ids = set(future_orders["customer_id"].astype(str))
        snapshot["churn"] = (
            ~snapshot["customer_id"].isin(future_customer_ids)
        ).astype(int)
        snapshot["snapshot_date"] = snapshot_date.strftime("%Y-%m-%d")

        # Customers without previous orders are represented with zero activity.
        zero_fill_columns = [
            "total_orders", "total_spend", "total_quantity",
            "cancelled_orders", "completed_orders", "cancel_rate",
            "distinct_product_categories",
        ]
        for column in zero_fill_columns:
            snapshot[column] = snapshot[column].fillna(0)

        numeric_columns = snapshot.select_dtypes(include=[np.number]).columns
        snapshot[numeric_columns] = snapshot[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        snapshot_frames.append(snapshot)

    if not snapshot_frames:
        raise ValueError("No customer snapshots could be generated.")

    result = pd.concat(snapshot_frames, ignore_index=True)
    output_path = OUTPUT_DIR / "customer_churn_training.csv"
    result.to_csv(output_path, index=False)

    report = {
        "target_definition": (
            "churn=1 when no order occurs in the 30 days immediately after "
            "snapshot_date; churn=0 otherwise."
        ),
        "feature_window": "Orders timestamped on or before snapshot_date only.",
        "label_window_days": 30,
        "snapshot_dates": [date.strftime("%Y-%m-%d") for date in snapshots],
        "rows": int(len(result)),
        "customers": int(result["customer_id"].nunique()),
        "churn_rate": float(result["churn"].mean()),
        "limitations": [
            "The synthetic generator does not supply a verified churn label.",
            "Churn is operationally defined as 30-day inactivity.",
            "Customers can appear in multiple monthly snapshots.",
            "The available synthetic history is short and may not represent real retention behaviour.",
        ],
    }
    (OUTPUT_DIR / "churn_target_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Created {len(result)} customer-snapshot rows.")
    print(f"Churn rate: {result['churn'].mean():.3f}")
    print(f"Saved: {output_path}")
    return result


if __name__ == "__main__":
    build_customer_snapshots()
