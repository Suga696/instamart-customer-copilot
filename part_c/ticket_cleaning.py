"""
Part C - Ticket cleaning
------------------------
Cleans the raw support-ticket export before PII masking and classification.

Input:
    data/raw/tickets.csv

Outputs:
    data/processed/part_c/cleaned_tickets.csv
    data/processed/part_c/ticket_cleaning_report.json

Important:
- The supplied raw CSV is never modified.
- Empty ticket descriptions are retained so the classification/routing stage
  can handle them according to the assignment rules.
- Long descriptions are not truncated.
- This module does not call the mock LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_c"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_tickets.csv"
REPORT_FILE = OUTPUT_DIR / "ticket_cleaning_report.json"

REQUIRED_COLUMNS = {
    "ticket_id",
    "customer_id",
    "order_id",
    "text",
    "true_category",
}

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}


def normalize_identifier(value):
    """Trim whitespace from IDs while preserving missing values."""
    if pd.isna(value):
        return pd.NA
    cleaned = str(value).strip()
    return cleaned if cleaned else pd.NA


def clean_ticket_text(value):
    """
    Normalize whitespace without changing the meaning of the ticket.

    PII is intentionally not masked here; pii_masking.py performs that step
    after this cleaning stage.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_category(value):
    """Normalize known category labels; leave unknown labels visible."""
    if pd.isna(value):
        return pd.NA

    category = str(value).strip().lower()
    return category if category else pd.NA


def clean_tickets(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Load and clean tickets, retaining empty and long descriptions."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Ticket file not found: {input_file}. "
            "Run generate_data.py from the repository root first."
        )

    tickets = pd.read_csv(input_file, dtype=str)

    missing_columns = REQUIRED_COLUMNS - set(tickets.columns)
    if missing_columns:
        raise ValueError(
            "tickets.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    original_rows = len(tickets)
    original_duplicate_ids = int(tickets["ticket_id"].duplicated().sum())

    # Standardize identifiers and text fields.
    for column in ("ticket_id", "customer_id", "order_id"):
        tickets[column] = tickets[column].map(normalize_identifier)

    tickets["text"] = tickets["text"].map(clean_ticket_text)
    tickets["true_category"] = tickets["true_category"].map(normalize_category)

    # Remove exact duplicate records first, then keep the first record for
    # any repeated ticket_id. Ticket IDs should uniquely identify a ticket.
    exact_duplicates_removed = int(tickets.duplicated().sum())
    tickets = tickets.drop_duplicates()

    duplicate_ticket_ids_removed = int(tickets["ticket_id"].duplicated().sum())
    tickets = tickets.drop_duplicates(subset=["ticket_id"], keep="first")

    # Keep rows with blank descriptions. They are valid test cases for
    # downstream fallback/empty-input handling.
    empty_description_count = int(tickets["text"].eq("").sum())
    long_description_count = int(
        tickets["text"].map(lambda value: len(value.split()) > 500).sum()
    )

    # Report unexpected labels but do not silently rewrite them.
    observed_categories = sorted(
        str(value)
        for value in tickets["true_category"].dropna().unique()
    )
    unexpected_categories = sorted(
        set(observed_categories) - ALLOWED_CATEGORIES - {"none"}
    )

    # A missing ticket ID cannot be used reliably by later stages.
    missing_ticket_id_count = int(tickets["ticket_id"].isna().sum())
    if missing_ticket_id_count:
        tickets = tickets[tickets["ticket_id"].notna()].copy()

    tickets = tickets.reset_index(drop=True)

    report = {
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "rows_before_cleaning": original_rows,
        "rows_after_cleaning": len(tickets),
        "exact_duplicate_rows_removed": exact_duplicates_removed,
        "duplicate_ticket_ids_present_before_cleaning": original_duplicate_ids,
        "duplicate_ticket_id_rows_removed": duplicate_ticket_ids_removed,
        "rows_removed_for_missing_ticket_id": missing_ticket_id_count,
        "empty_descriptions_retained": empty_description_count,
        "long_descriptions_over_500_words_retained": long_description_count,
        "observed_true_categories": observed_categories,
        "unexpected_true_categories": unexpected_categories,
        "pii_masking_performed": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(OUTPUT_FILE, index=False)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Cleaned tickets saved to: {OUTPUT_FILE}")
    print(f"Cleaning report saved to: {REPORT_FILE}")
    print(f"Rows before cleaning: {original_rows}")
    print(f"Rows after cleaning:  {len(tickets)}")
    print(f"Empty descriptions retained: {empty_description_count}")
    print(f"Long descriptions retained: {long_description_count}")

    return tickets


if __name__ == "__main__":
    clean_tickets()
