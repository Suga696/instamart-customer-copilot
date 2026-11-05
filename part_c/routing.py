"""
Part C - Ticket routing
----------------------
Joins classified tickets with Part B customer cluster assignments and applies
the routing/SLA rules supported by the supplied project policy text.

Inputs:
    data/processed/part_c/classified_tickets.csv
    data/processed/part_b/customer_segments.csv

Outputs:
    data/processed/part_c/routed_tickets.csv
    data/processed/part_c/routing_report.json

Important segment note:
Part B currently creates numeric KMeans cluster labels (0, 1, ...). These
numbers are arbitrary and must NOT be treated as Premium/Regular labels.
Set PREMIUM_CLUSTER_IDS only after the assignment brief or segment profile
explicitly identifies which cluster represents Premium customers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKETS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "part_c" / "classified_tickets.csv"
)
SEGMENTS_FILE = (
    PROJECT_ROOT / "data" / "processed" / "part_b" / "customer_segments.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_c"
OUTPUT_FILE = OUTPUT_DIR / "routed_tickets.csv"
REPORT_FILE = OUTPUT_DIR / "routing_report.json"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Configure only when the assignment brief or an approved segment analysis
# explicitly maps a numeric KMeans cluster to the Premium customer segment.
# Example after confirmation: PREMIUM_CLUSTER_IDS = {2}
PREMIUM_CLUSTER_IDS = set()

CATEGORY_QUEUE = {
    "billing": "Billing Support",
    "delivery": "Delivery Support",
    "technical": "Technical Support",
    "fraud": "Security Team",
}


def load_inputs(
    tickets_file: Path = TICKETS_FILE,
    segments_file: Path = SEGMENTS_FILE,
):
    """Load ticket predictions and customer cluster assignments."""
    if not tickets_file.exists():
        raise FileNotFoundError(
            f"Classified tickets not found: {tickets_file}. "
            "Run classification.py first."
        )

    if not segments_file.exists():
        raise FileNotFoundError(
            f"Customer segments not found: {segments_file}. "
            "Run Part B segmentation.py first."
        )

    tickets = pd.read_csv(tickets_file, dtype=str, keep_default_na=False)
    segments = pd.read_csv(segments_file, dtype=str, keep_default_na=False)

    required_ticket_columns = {"ticket_id", "customer_id", "predicted_category"}
    required_segment_columns = {"customer_id", "cluster"}

    missing_ticket_columns = required_ticket_columns - set(tickets.columns)
    missing_segment_columns = required_segment_columns - set(segments.columns)

    if missing_ticket_columns:
        raise ValueError(
            "classified_tickets.csv is missing columns: "
            + ", ".join(sorted(missing_ticket_columns))
        )

    if missing_segment_columns:
        raise ValueError(
            "customer_segments.csv is missing columns: "
            + ", ".join(sorted(missing_segment_columns))
        )

    if segments["customer_id"].duplicated().any():
        raise ValueError(
            "customer_segments.csv must contain only one row per customer_id."
        )

    return tickets, segments


def route_category(category):
    """Return the support queue for a validated predicted category."""
    normalized = str(category).strip().lower()

    if normalized == "needs_review":
        return "Manual Review"
    if normalized == "unclassified":
        return "Manual Review"
    if normalized not in ALLOWED_CATEGORIES:
        return "Manual Review"

    return CATEGORY_QUEUE[normalized]


def route_ticket(row):
    """
    Apply routing and policy-backed response targets.

    Policy facts used:
    - Fraud/unauthorized-transaction tickets: high priority, Security Team,
      response target within 2 hours.
    - Premium customers: response target within 1 hour.
    - All other customers: response target within 24 hours.

    Fraud takes precedence over the general premium/non-premium response rule.
    """
    category = str(row["predicted_category"]).strip().lower()
    cluster_value = str(row.get("cluster", "")).strip()

    try:
        cluster_id = int(float(cluster_value))
    except (TypeError, ValueError):
        cluster_id = None

    is_premium = (
        cluster_id is not None and cluster_id in PREMIUM_CLUSTER_IDS
    )

    queue = route_category(category)

    if category == "fraud":
        priority = "HIGH"
        response_sla = "2 hours"
        routing_reason = "Fraud category: send to Security Team"
    elif category in ALLOWED_CATEGORIES:
        priority = "STANDARD"
        if is_premium:
            response_sla = "1 hour"
            routing_reason = "Premium customer response target"
        else:
            response_sla = "24 hours"
            routing_reason = "Standard customer response target"
    else:
        priority = "REVIEW"
        response_sla = "Manual review required"
        routing_reason = "Ticket could not be confidently classified"

    return pd.Series({
        "customer_cluster": cluster_value,
        "customer_tier": "Premium" if is_premium else "Not identified as Premium",
        "assigned_queue": queue,
        "priority": priority,
        "response_sla": response_sla,
        "routing_reason": routing_reason,
    })


def run_routing(
    tickets_file: Path = TICKETS_FILE,
    segments_file: Path = SEGMENTS_FILE,
) -> pd.DataFrame:
    """Join customer clusters, route tickets and save the routing report."""
    tickets, segments = load_inputs(tickets_file, segments_file)

    routed = tickets.merge(
        segments[["customer_id", "cluster"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_customers = int((routed["_merge"] == "left_only").sum())
    routed = routed.drop(columns=["_merge"])

    route_columns = routed.apply(route_ticket, axis=1)
    routed = pd.concat([routed, route_columns], axis=1)

    # Customers without a Part B match should be visible and reviewed.
    missing_cluster_mask = routed["cluster"].eq("")
    routed.loc[missing_cluster_mask, "assigned_queue"] = "Manual Review"
    routed.loc[missing_cluster_mask, "priority"] = "REVIEW"
    routed.loc[missing_cluster_mask, "response_sla"] = "Manual review required"
    routed.loc[missing_cluster_mask, "routing_reason"] = (
        "No customer segment assignment found"
    )
    routed.loc[missing_cluster_mask, "customer_tier"] = "Unknown"

    queue_counts = {
        str(key): int(value)
        for key, value in routed["assigned_queue"].value_counts().items()
    }
    priority_counts = {
        str(key): int(value)
        for key, value in routed["priority"].value_counts().items()
    }
    sla_counts = {
        str(key): int(value)
        for key, value in routed["response_sla"].value_counts().items()
    }

    report = {
        "tickets_processed": int(len(routed)),
        "customers_without_segment_match": unmatched_customers,
        "premium_cluster_ids_configured": sorted(PREMIUM_CLUSTER_IDS),
        "queue_counts": queue_counts,
        "priority_counts": priority_counts,
        "response_sla_counts": sla_counts,
        "fraud_tickets_routed_to_security": int(
            (
                routed["predicted_category"].str.lower().eq("fraud")
                & routed["assigned_queue"].eq("Security Team")
            ).sum()
        ),
        "manual_review_tickets": int(
            routed["assigned_queue"].eq("Manual Review").sum()
        ),
        "policy_rules_applied": {
            "fraud": "High priority; Security Team; response within 2 hours",
            "premium": "Response within 1 hour when Premium cluster is explicitly configured",
            "other_customers": "Response within 24 hours",
        },
        "segment_mapping_warning": (
            "KMeans cluster IDs are arbitrary. Premium cluster mapping is not "
            "assumed; configure PREMIUM_CLUSTER_IDS only when confirmed."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    routed.to_csv(OUTPUT_FILE, index=False)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Routed tickets saved to: {OUTPUT_FILE}")
    print(f"Routing report saved to: {REPORT_FILE}")
    print(f"Tickets processed: {len(routed)}")
    print(f"Queue counts: {queue_counts}")
    print(f"Priority counts: {priority_counts}")
    print(f"Customers without segment match: {unmatched_customers}")

    return routed


if __name__ == "__main__":
    run_routing()
