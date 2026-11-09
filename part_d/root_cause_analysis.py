"""Analyse delivery and complaint patterns; do not infer causation."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "part_d"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv(PROC / "orders_clean.csv")
    customers = pd.read_csv(PROC / "customers_clean.csv")
    tickets = pd.read_csv(PROC / "part_c" / "classified_tickets.csv")
    orders["order_ts"] = pd.to_datetime(orders.order_ts, errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(orders.delivery_minutes, errors="coerce")
    orders["order_amount"] = pd.to_numeric(orders.order_amount, errors="coerce")
    orders = orders.merge(customers[["customer_id","city"]], on="customer_id", how="left")
    orders["month"] = orders.order_ts.dt.to_period("M").astype(str)
    orders["hour"] = orders.order_ts.dt.hour
    city = orders.groupby("city").agg(
        order_count=("order_id","nunique"),
        avg_delivery_minutes=("delivery_minutes","mean"),
        median_delivery_minutes=("delivery_minutes","median"),
        delayed_over_30_min=("delivery_minutes", lambda x: int((x > 30).sum())),
        missing_delivery_time=("delivery_minutes", lambda x: int(x.isna().sum())),
    ).reset_index()
    city["delay_rate_pct"] = (
        city.delayed_over_30_min / city.order_count.clip(lower=1) * 100
    ).round(2)
    city.to_csv(OUT / "delivery_city_analysis.csv", index=False)
    monthly = orders.groupby(["month","city"]).agg(
        order_count=("order_id","nunique"),
        avg_delivery_minutes=("delivery_minutes","mean"),
        delayed_over_30_min=("delivery_minutes", lambda x: int((x > 30).sum())),
    ).reset_index()
    monthly.to_csv(OUT / "delivery_monthly_analysis.csv", index=False)
    hourly = orders.groupby("hour").agg(
        order_count=("order_id","nunique"),
        avg_delivery_minutes=("delivery_minutes","mean"),
        delayed_over_30_min=("delivery_minutes", lambda x: int((x > 30).sum())),
    ).reset_index()
    hourly.to_csv(OUT / "delivery_hourly_analysis.csv", index=False)
    delivery_tickets = tickets[tickets.predicted_category.astype(str).str.lower().eq("delivery")]
    top_city = city.sort_values("delay_rate_pct", ascending=False).head(1)
    top_hour = hourly.sort_values("avg_delivery_minutes", ascending=False).head(1)
    report = {
        "delivery_order_count": int(len(orders)),
        "delivery_ticket_count": int(len(delivery_tickets)),
        "city_metrics": city.to_dict(orient="records"),
        "highest_delay_rate_city": top_city.iloc[0].to_dict() if len(top_city) else None,
        "highest_average_delivery_hour": top_hour.iloc[0].to_dict() if len(top_hour) else None,
        "interpretation": "These are descriptive associations in synthetic order and ticket data. City/hour patterns do not establish operational causes.",
        "data_limitations": [
            "No warehouse, delivery zone, rider workload or weather fields are supplied.",
            "Delivery delay threshold of more than 30 minutes is a descriptive comparison against the policy's standard promise; rain exceptions are not represented in the data.",
            "Support-ticket category is predicted by the mock classifier and is not independent ground truth.",
        ],
    }
    (OUT / "root_cause_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Delivery pattern analysis complete; no causal claims made.")

if __name__ == "__main__":
    main()
