"""Create the consolidated Customer Intelligence Report."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "part_b"


def main():
    segments_path = OUTPUT_DIR / "customer_segments.csv"
    profiles_path = OUTPUT_DIR / "segment_profile.csv"
    segmentation_metrics_path = OUTPUT_DIR / "segmentation_metrics.json"
    churn_metrics_path = OUTPUT_DIR / "churn_model_comparison.json"

    required_files = [
        segments_path,
        profiles_path,
        segmentation_metrics_path,
        churn_metrics_path,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Required report inputs are missing. Run the Part B workflow first:\\n"
            + "\\n".join(missing_files)
        )

    segments = pd.read_csv(segments_path)
    profiles = pd.read_csv(profiles_path)

    segmentation_metrics = json.loads(
        segmentation_metrics_path.read_text(encoding="utf-8")
    )
    churn_metrics = json.loads(churn_metrics_path.read_text(encoding="utf-8"))

    report = {
        "title": "Customer Intelligence Report",
        "segmentation": {
            "customers_segmented": int(segments["customer_id"].nunique()),
            "cluster_count": int(segments["cluster"].nunique()),
            "metrics": segmentation_metrics,
            "cluster_profiles": profiles.to_dict(orient="records"),
            "interpretation_caution": (
                "K-Means cluster IDs are arbitrary. Interpret each segment using "
                "its original feature averages; do not assume Premium or Regular "
                "labels unless an authoritative mapping is supplied."
            ),
        },
        "churn": churn_metrics,
        "business_use": (
            "Use customer profiles and out-of-time churn risk estimates to design "
            "measurable retention experiments."
        ),
        "limitations": [
            "The churn target is derived from no order in the next 30 days; it is not a supplied verified churn label.",
            "The synthetic order history covers a limited period.",
            "Repeated customers occur across monthly snapshots in the time-based split.",
            "Predictive associations do not establish why a customer stopped ordering.",
        ],
    }

    output_path = OUTPUT_DIR / "customer_intelligence_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Customer Intelligence Report generated.")
    print(f"Customers segmented: {report['segmentation']['customers_segmented']}")
    print(f"Clusters: {report['segmentation']['cluster_count']}")
    print(f"Selected churn model: {churn_metrics['selected_model_by_pr_auc']}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
