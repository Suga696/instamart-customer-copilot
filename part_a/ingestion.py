import json
from pathlib import Path

import pandas as pd


# Resolve paths from the repository location, not from the terminal's
# current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
REPORT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = REPORT_DIR / "ingestion_report.json"


# Expected column names are taken from the supplied generate_data.py.
CSV_FILES = {
    "customers.csv": [
        "customer_id", "city", "signup_date", "age", "gender"
    ],
    "products.csv": [
        "product_id", "product_name", "category", "price_inr"
    ],
    "orders.csv": [
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes"
    ],
    "tickets.csv": [
        "ticket_id", "customer_id", "order_id", "text", "true_category"
    ],
    "policy_eval.csv": [
        "question", "expected_doc", "expected_fact"
    ],
    "agent_scenarios.csv": [
        "scenario_id", "customer_id", "message", "expected_outcome",
        "expected_refund_inr"
    ],
}

POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
    "escalation_policy.txt",
]


def load_csv_files():
    """Read each expected CSV and validate its column names."""
    datasets = {}

    for filename, expected_columns in CSV_FILES.items():
        file_path = RAW_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing input file: {file_path}\n"
                "Run 'python generate_data.py' from the repository root first."
            )

        dataframe = pd.read_csv(file_path)

        missing_columns = [
            column for column in expected_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{filename} is missing expected columns: {missing_columns}"
            )

        datasets[filename] = dataframe

    return datasets


def profile_dataframe(dataframe):
    """Return basic row, column, missing-value and duplicate statistics."""
    missing_by_column = {
        column: int(count)
        for column, count in dataframe.isna().sum().items()
    }

    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": list(dataframe.columns),
        "exact_duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values_by_column": missing_by_column,
        "total_missing_values": int(dataframe.isna().sum().sum()),
    }


def check_policy_files():
    """Check that all policy text files exist and are not empty."""
    policy_status = {}

    for filename in POLICY_FILES:
        file_path = POLICY_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Missing policy file: {file_path}")

        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Policy file is empty: {file_path}")

        policy_status[filename] = {
            "exists": True,
            "characters": len(content),
        }

    return policy_status


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART A: RAW DATA INGESTION")
    print("=" * 65)

    datasets = load_csv_files()
    report = {
        "project": "Instamart Customer Intelligence & Support Copilot",
        "stage": "Part A - Raw data ingestion and initial profiling",
        "datasets": {},
        "policy_files": check_policy_files(),
    }

    for filename, dataframe in datasets.items():
        profile = profile_dataframe(dataframe)
        report["datasets"][filename] = profile

        print(f"\n{filename}")
        print("-" * 45)
        print(f"Rows:                 {profile['rows']}")
        print(f"Columns:              {profile['columns']}")
        print(f"Exact duplicate rows: {profile['exact_duplicate_rows']}")
        print(f"Total missing values: {profile['total_missing_values']}")

        missing_values = {
            column: count
            for column, count in profile["missing_values_by_column"].items()
            if count > 0
        }
        if missing_values:
            print(f"Missing by column:    {missing_values}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPolicy files checked:")
    for filename, details in report["policy_files"].items():
        print(f"- {filename}: {details['characters']} characters")

    print("\n" + "=" * 65)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print(f"Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Raw source files were not modified.")
    print("=" * 65)


if __name__ == "__main__":
    main()
