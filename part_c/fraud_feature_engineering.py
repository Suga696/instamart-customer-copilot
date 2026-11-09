"""Build order-level behavioural indicators for anomaly screening."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "part_c"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    orders = pd.read_csv(PROC / "orders_clean.csv")
    customers = pd.read_csv(PROC / "customers_clean.csv")
    tickets_path = OUT / "classified_tickets.csv"
    orders["customer_id"] = orders.customer_id.astype(str)
    for c in ["quantity", "order_amount", "delivery_minutes", "discount_pct"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")
    orders["order_ts"] = pd.to_datetime(orders.order_ts, errors="coerce")
    customer_stats = orders.groupby("customer_id").agg(
        customer_order_count=("order_id", "nunique"),
        customer_mean_order_amount=("order_amount", "mean"),
        customer_std_order_amount=("order_amount", "std"),
        customer_cancel_count=("status", lambda x: int((x == "cancelled").sum())),
        customer_completed_count=("status", lambda x: int((x == "completed").sum())),
    )
    customer_stats["customer_cancel_rate"] = (
        customer_stats.customer_cancel_count
        / orders.groupby("customer_id").order_id.nunique().replace(0, np.nan)
    )
    result = orders.merge(customer_stats, left_on="customer_id", right_index=True, how="left")
    result["amount_vs_customer_mean"] = (
        result.order_amount - result.customer_mean_order_amount
    )
    result["amount_z_like"] = (
        result.amount_vs_customer_mean
        / result.customer_std_order_amount.replace(0, np.nan)
    )
    if tickets_path.exists():
        tickets = pd.read_csv(tickets_path)
        fraud_counts = tickets.assign(
            is_fraud=tickets.predicted_category.astype(str).str.lower().eq("fraud")
        ).groupby("customer_id").is_fraud.sum()
        result["customer_fraud_ticket_count"] = result.customer_id.map(fraud_counts).fillna(0)
    else:
        result["customer_fraud_ticket_count"] = 0
    result["is_completed"] = result.status.astype(str).str.lower().eq("completed").astype(int)
    result["is_cancelled"] = result.status.astype(str).str.lower().eq("cancelled").astype(int)
    keep = [
        "order_id", "customer_id", "product_id", "order_ts", "status",
        "order_amount", "quantity", "delivery_minutes", "discount_pct",
        "customer_order_count", "customer_mean_order_amount",
        "customer_cancel_rate", "amount_vs_customer_mean", "amount_z_like",
        "customer_fraud_ticket_count", "is_completed", "is_cancelled",
    ]
    result[keep].to_csv(OUT / "fraud_features.csv", index=False)
    report = {
        "rows": int(len(result)),
        "feature_columns": keep[5:],
        "available_sources": ["cleaned orders", "cleaned customers"]
            + (["classified support tickets"] if tickets_path.exists() else []),
        "limitations": [
            "No verified transaction-level fraud labels exist in the supplied synthetic data.",
            "Customer aggregates use the full observation period and are retrospective descriptive indicators, not leakage-safe prospective predictions.",
            "A fraud-labelled support ticket is a complaint category, not proof that the linked transaction is fraudulent.",
        ],
    }
    (OUT / "fraud_feature_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Fraud indicators created for {len(result)} orders.")

if __name__ == "__main__":
    main()
