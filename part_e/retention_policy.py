"""Create evidence-linked retention recommendations from customer segments."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "part_e"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(PROC / "part_b" / "customer_segments.csv")
    prof = pd.read_csv(PROC / "part_b" / "segment_profile.csv")
    # Use profile values for interpretation; cluster IDs remain arbitrary.
    recommendations = []
    for _, r in prof.iterrows():
        cluster = int(r["cluster"])
        spend = float(r.get("total_order_amount", 0) or 0)
        orders = float(r.get("total_orders", 0) or 0)
        discount = float(r.get("average_discount_pct", 0) or 0)
        recency = float(r.get("days_since_last_order", 0) or 0)
        if spend >= prof["total_order_amount"].quantile(.75):
            action = "Consider loyalty recognition and service benefits; validate incremental margin before rewards."
            group = "Higher-spend segment"
        elif recency >= prof["days_since_last_order"].median():
            action = "Test a targeted re-engagement message or incentive; measure incremental repeat orders."
            group = "Potentially declining engagement"
        elif discount >= prof["average_discount_pct"].median():
            action = "Use personalised promotions with margin monitoring; avoid blanket discounts."
            group = "Discount-responsive segment"
        elif orders <= prof["total_orders"].median():
            action = "Provide onboarding and product-discovery prompts; monitor second-order conversion."
            group = "Lower-frequency / newer segment"
        else:
            action = "Maintain relevant engagement and monitor repeat-purchase behaviour."
            group = "Steady-engagement segment"
        recommendations.append({
            "cluster_id": cluster, "descriptive_group": group,
            "customers": int((seg.cluster == cluster).sum()),
            "avg_total_spend_inr": spend, "avg_total_orders": orders,
            "avg_days_since_last_order": recency,
            "avg_discount_pct": discount, "recommended_action": action,
            "measurement": "Compare repeat-order rate and contribution margin against a holdout group.",
        })
    pd.DataFrame(recommendations).to_csv(OUT / "retention_recommendations.csv", index=False)
    (OUT / "retention_policy.json").write_text(json.dumps({
        "policy": recommendations,
        "caveat": "Cluster IDs are arbitrary and segment names are descriptive interpretations, not source-provided labels.",
    }, indent=2), encoding="utf-8")
    print("Retention policy recommendations generated.")

if __name__ == "__main__":
    main()
