"""
Part A - Consolidated data quality report.

Run from the repository root, after generating data and completing Part A:
    python part_a/data_quality_report.py

Required inputs:
    data/raw/customers.csv
    data/raw/products.csv
    data/raw/orders.csv
    data/processed/customers_clean.csv
    data/processed/products_clean.csv
    data/processed/orders_clean.csv
    data/processed/ingestion_report.json
    data/processed/cleaning_report.json
    data/processed/validation_report.json

Outputs:
    data/processed/data_quality_report.json
    data/processed/data_quality_summary.csv

The report consolidates the existing ingestion, cleaning and validation
artifacts and independently calculates before/after row counts, missing
values, exact duplicate rows and key-duplicate counts. It does not modify
source data or replace the detailed validation checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "customers": {
        "raw_file": "customers.csv",
        "clean_file": "customers_clean.csv",
        "primary_key": "customer_id",
        "description": "Customer master data and demographic attributes.",
        "columns": {
            "customer_id": "Unique customer identifier.",
            "city": "Customer city.",
            "signup_date": "Customer signup date.",
            "age": "Customer age in years; invalid ages are represented as missing.",
            "gender": "Standardized customer gender category.",
        },
    },
    "products": {
        "raw_file": "products.csv",
        "clean_file": "products_clean.csv",
        "primary_key": "product_id",
        "description": "Product master data and listed prices.",
        "columns": {
            "product_id": "Unique product identifier.",
            "product_name": "Product display name.",
            "category": "Product category.",
            "price_inr": "Product price in Indian rupees.",
        },
    },
    "orders": {
        "raw_file": "orders.csv",
        "clean_file": "orders_clean.csv",
        "primary_key": "order_id",
        "description": "Customer order and delivery transaction records.",
        "columns": {
            "order_id": "Unique order identifier.",
            "customer_id": "Customer who placed the order.",
            "product_id": "Product included in the order.",
            "quantity": "Number of product units ordered.",
            "order_ts": "Order timestamp.",
            "status": "Order status.",
            "discount_pct": "Discount percentage applied.",
            "order_amount": "Order amount in Indian rupees.",
            "delivery_minutes": "Delivery duration in minutes.",
        },
    },
}

REPORT_INPUTS = {
    "ingestion": PROCESSED_DIR / "ingestion_report.json",
    "cleaning": PROCESSED_DIR / "cleaning_report.json",
    "validation": PROCESSED_DIR / "validation_report.json",
}

JSON_OUTPUT = PROCESSED_DIR / "data_quality_report.json"
CSV_OUTPUT = PROCESSED_DIR / "data_quality_summary.csv"


def read_json(path: Path, label: str) -> dict:
    """Read a required JSON report with a useful error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required {label} not found: {path}. "
            "Run the preceding Part A scripts first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}. "
            "Run generate_data.py and Part A cleaning.py first."
        )
    return pd.read_csv(path)


def dataframe_quality(dataframe: pd.DataFrame, primary_key: str) -> dict:
    """Calculate general quality metrics for one dataset."""
    missing_by_column = {
        str(column): int(count)
        for column, count in dataframe.isna().sum().items()
    }
    key_missing = (
        int(dataframe[primary_key].isna().sum())
        if primary_key in dataframe.columns else None
    )
    key_duplicates = (
        int(dataframe[primary_key].dropna().duplicated().sum())
        if primary_key in dataframe.columns else None
    )

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": [str(column) for column in dataframe.columns],
        "exact_duplicate_rows": int(dataframe.duplicated().sum()),
        "missing_values_total": int(dataframe.isna().sum().sum()),
        "missing_values_by_column": missing_by_column,
        "primary_key": primary_key,
        "primary_key_missing": key_missing,
        "primary_key_duplicates": key_duplicates,
    }


def load_validation_summary(validation_report: dict) -> dict:
    """Extract validation totals without changing the original report."""
    checks = validation_report.get("checks", [])
    if not isinstance(checks, list):
        checks = []

    passed = sum(
        1 for item in checks
        if str(item.get("status", "")).upper() == "PASS"
    )
    failed = sum(
        1 for item in checks
        if str(item.get("status", "")).upper() == "FAIL"
    )

    # Preserve the original report's summary fields as well, if available.
    summary = {
        "checks_total": len(checks),
        "checks_passed": passed,
        "checks_failed": failed,
        "checks": checks,
    }
    for key in ("status", "overall_status", "passed", "failed", "total_checks"):
        if key in validation_report:
            summary[f"source_{key}"] = validation_report[key]
    return summary


def build_report() -> tuple[dict, pd.DataFrame]:
    """Build consolidated JSON and tabular data-quality summaries."""
    ingestion_report = read_json(REPORT_INPUTS["ingestion"], "ingestion report")
    cleaning_report = read_json(REPORT_INPUTS["cleaning"], "cleaning report")
    validation_report = read_json(REPORT_INPUTS["validation"], "validation report")

    dataset_results = {}
    summary_rows = []

    for dataset_name, config in DATASETS.items():
        raw_df = read_csv(RAW_DIR / config["raw_file"])
        clean_df = read_csv(PROCESSED_DIR / config["clean_file"])

        raw_quality = dataframe_quality(raw_df, config["primary_key"])
        clean_quality = dataframe_quality(clean_df, config["primary_key"])

        rows_removed = raw_quality["row_count"] - clean_quality["row_count"]
        missing_reduction = (
            raw_quality["missing_values_total"]
            - clean_quality["missing_values_total"]
        )

        dataset_results[dataset_name] = {
            "description": config["description"],
            "data_dictionary": config["columns"],
            "raw": raw_quality,
            "cleaned": clean_quality,
            "change_summary": {
                "rows_removed": int(rows_removed),
                "missing_values_reduced_by": int(missing_reduction),
                "exact_duplicate_rows_removed": int(
                    raw_quality["exact_duplicate_rows"]
                    - clean_quality["exact_duplicate_rows"]
                ),
                "primary_key_duplicates_removed": (
                    int(
                        raw_quality["primary_key_duplicates"]
                        - clean_quality["primary_key_duplicates"]
                    )
                    if raw_quality["primary_key_duplicates"] is not None
                    and clean_quality["primary_key_duplicates"] is not None
                    else None
                ),
            },
        }

        summary_rows.append({
            "dataset": dataset_name,
            "raw_rows": raw_quality["row_count"],
            "cleaned_rows": clean_quality["row_count"],
            "rows_removed": rows_removed,
            "raw_missing_values": raw_quality["missing_values_total"],
            "cleaned_missing_values": clean_quality["missing_values_total"],
            "raw_exact_duplicates": raw_quality["exact_duplicate_rows"],
            "cleaned_exact_duplicates": clean_quality["exact_duplicate_rows"],
            "raw_primary_key_duplicates": raw_quality["primary_key_duplicates"],
            "cleaned_primary_key_duplicates": clean_quality["primary_key_duplicates"],
        })

    report = {
        "project": "Swiggy Instamart Capstone",
        "report_name": "Part A - Consolidated Data Quality Report",
        "purpose": (
            "Summarise raw-to-cleaned data quality changes and consolidate "
            "the ingestion, cleaning and validation outputs."
        ),
        "datasets": dataset_results,
        "cleaning_actions_recorded": cleaning_report,
        "ingestion_profile": ingestion_report,
        "validation_summary": load_validation_summary(validation_report),
        "interpretation_notes": [
            "A reduction in missing values is not guaranteed; cleaning may preserve missing values to avoid silently discarding records.",
            "Rows removed can include exact duplicates and repeated order IDs resolved by the cleaning rules.",
            "For orders, the cleaning process resolves repeated order IDs by preferring completed/cancelled records over pending records and then the latest timestamp.",
            "Invalid ages, prices, quantities, discounts, amounts and negative delivery durations are set to missing by the current cleaning implementation where applicable.",
            "Validation checks and their detailed outcomes are retained from validation_report.json.",
            "A passing validation check does not mean every field is complete; expected missing values are separately reported.",
        ],
        "data_dictionary": {
            name: config["columns"] for name, config in DATASETS.items()
        },
        "source_reports": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in REPORT_INPUTS.items()
        },
        "outputs": [
            str(JSON_OUTPUT.relative_to(PROJECT_ROOT)),
            str(CSV_OUTPUT.relative_to(PROJECT_ROOT)),
        ],
    }

    summary_df = pd.DataFrame(summary_rows)
    return report, summary_df


def main() -> None:
    """Generate the consolidated data quality report."""
    print("=" * 68)
    print("SWIGGY INSTAMART CAPSTONE - PART A: DATA QUALITY REPORT")
    print("=" * 68)

    report, summary_df = build_report()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    JSON_OUTPUT.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_df.to_csv(CSV_OUTPUT, index=False)

    print("\nRaw-to-cleaned dataset summary")
    print(summary_df.to_string(index=False))
    validation = report["validation_summary"]
    print("\nValidation summary")
    print(f"Checks found: {validation['checks_total']}")
    print(f"Passed:       {validation['checks_passed']}")
    print(f"Failed:       {validation['checks_failed']}")
    print(f"\nJSON report: {JSON_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"CSV summary: {CSV_OUTPUT.relative_to(PROJECT_ROOT)}")
    print("\nDATA QUALITY REPORT GENERATED")


if __name__ == "__main__":
    main()
