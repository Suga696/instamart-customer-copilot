"""Produce human-review investigation notes for flagged anomalous orders."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "part_d"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anomalies = pd.read_csv(PROC / "part_c" / "fraud_anomaly_results.csv")
    features = pd.read_csv(PROC / "part_c" / "fraud_features.csv")
    tickets = pd.read_csv(PROC / "part_c" / "classified_tickets.csv")
    flagged = anomalies[anomalies.anomaly_flag.astype(str).str.lower().isin(["true","1"])].copy()
    details = flagged.merge(features, on=["order_id","customer_id","order_ts","status"], how="left")
    ticket_groups = tickets.groupby("customer_id")
    notes = []
    for row in details.itertuples():
        customer_tickets = ticket_groups.get_group(row.customer_id) if row.customer_id in ticket_groups.groups else pd.DataFrame()
        cats = customer_tickets.predicted_category.value_counts().to_dict() if len(customer_tickets) else {}
        notes.append({
            "order_id": row.order_id,
            "customer_id": row.customer_id,
            "risk_category": row.review_priority,
            "anomaly_score": float(row.anomaly_score),
            "main_indicators": {
                "order_amount": None if pd.isna(row.order_amount) else float(row.order_amount),
                "quantity": None if pd.isna(row.quantity) else float(row.quantity),
                "delivery_minutes": None if pd.isna(row.delivery_minutes) else float(row.delivery_minutes),
                "customer_cancel_rate": None if pd.isna(row.customer_cancel_rate) else float(row.customer_cancel_rate),
                "customer_fraud_ticket_count": int(row.customer_fraud_ticket_count),
            },
            "support_ticket_category_counts": cats,
            "recommended_action": "Human investigation and verification; do not automatically reject a refund or penalise the customer.",
            "explanation": "The order was flagged as statistically unusual by Isolation Forest. Anomaly status is not proof of fraud.",
        })
    pd.DataFrame(notes).to_csv(OUT / "fraud_investigation_notes.csv", index=False)
    report = {
        "flagged_orders_reviewed": len(notes),
        "investigation_notes": notes,
        "method": "Rule-based evidence assembly from Isolation Forest flags, transaction indicators and customer ticket-category counts.",
        "llm_limitation": "The supplied mock LLM has no fraud-investigation mode; these are structured evidence notes, not LLM-generated conclusions.",
        "human_review_required": True,
    }
    (OUT / "fraud_investigation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {len(notes)} human-review investigation notes.")

if __name__ == "__main__":
    main()
