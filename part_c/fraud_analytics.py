"""Generate the consolidated Fraud Analytics Report.

Run after fraud_detection.py and part_d/fraud_investigation_assistant.py.
Uses only Python's standard library plus pandas (already permitted by the project).
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PART_C = ROOT / "data" / "processed" / "part_c"
PART_D = ROOT / "data" / "processed" / "part_d"


def read_json(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the preceding pipeline step first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    detection = read_json(PART_C / "fraud_detection_report.json", "fraud detection report")
    investigation = read_json(
        PART_D / "fraud_investigation_report.json", "fraud investigation report"
    )
    results_path = PART_C / "fraud_anomaly_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing anomaly results: {results_path}. Run part_c/fraud_detection.py first."
        )

    results = pd.read_csv(results_path)
    flagged = results[results["anomaly_flag"].astype(str).str.lower().isin(["true", "1"])]
    priority_counts = (
        flagged["review_priority"].value_counts().to_dict()
        if not flagged.empty else {}
    )
    status_counts = (
        flagged["status"].fillna("unknown").astype(str).value_counts().to_dict()
        if not flagged.empty else {}
    )

    report = {
        "title": "Fraud Analytics Report",
        "detection": detection,
        "summary": {
            "orders_scored": int(len(results)),
            "orders_flagged_for_review": int(len(flagged)),
            "flag_rate": float(len(flagged) / len(results)) if len(results) else 0.0,
            "flagged_review_priority_counts": {
                str(key): int(value) for key, value in priority_counts.items()
            },
            "flagged_order_status_counts": {
                str(key): int(value) for key, value in status_counts.items()
            },
        },
        "investigation": {
            "flagged_orders_reviewed": int(investigation.get("flagged_orders_reviewed", 0)),
            "human_review_required": bool(investigation.get("human_review_required", True)),
            "method": investigation.get("method", "Not specified"),
            "llm_limitation": investigation.get("llm_limitation", "Not specified"),
        },
        "conclusion": (
            "The anomaly model prioritises unusual orders for human review. "
            "An anomaly is not proof of fraud, and no customer should be penalised "
            "solely because an order was flagged."
        ),
        "limitations": [
            "The supplied synthetic data contains no verified transaction-level fraud outcome labels.",
            "Supervised fraud model performance metrics such as precision, recall and PR-AUC cannot be calculated reliably.",
            "Isolation Forest identifies unusual patterns, not confirmed fraud.",
            "Customer-level aggregate indicators are retrospective and should not be interpreted as prospective evidence.",
            "Investigation notes support human review and are not final fraud determinations.",
        ],
    }

    output_path = PART_C / "fraud_analytics_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Fraud Analytics Report generated: {output_path}")


if __name__ == "__main__":
    main()
