from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "part_c" / "cleaned_tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_c"
OUTPUT_FILE = OUTPUT_DIR / "pii_masked_tickets.csv"
REPORT_FILE = OUTPUT_DIR / "pii_masking_report.json"

EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)

# Supports common Indian mobile formats:
# 9876543210, 98765 43210, 98765-43210, 987-654-3210,
# +91 9876543210, +91-98765-43210 and 09876543210.
PHONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\+?91[\s.-]?)?"
    r"(?:0[\s.-]?)?"
    r"(?:[6-9]\d{4}[\s.-]?\d{5}|[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4})"
    r"(?![A-Za-z0-9])"
)

EMAIL_TOKEN = "[EMAIL_REDACTED]"
PHONE_TOKEN = "[PHONE_REDACTED]"


def mask_pii(value):
    """
    Mask email addresses and Indian mobile numbers in a text value.

    Returns (masked_text, email_count, phone_count).
    """
    if pd.isna(value):
        return "", 0, 0

    text = str(value)
    text, email_count = EMAIL_PATTERN.subn(EMAIL_TOKEN, text)
    text, phone_count = PHONE_PATTERN.subn(PHONE_TOKEN, text)
    return text, email_count, phone_count


def mask_ticket_file(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Mask PII in cleaned tickets and write the output and audit report."""
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Cleaned ticket file not found: {input_file}. "
            "Run python part_c/ticket_cleaning.py first."
        )

    tickets = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    required_columns = {"ticket_id", "customer_id", "order_id", "text"}
    missing_columns = required_columns - set(tickets.columns)
    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    masked_results = tickets["text"].map(mask_pii)
    tickets["text"] = masked_results.map(lambda result: result[0])

    total_emails = int(masked_results.map(lambda result: result[1]).sum())
    total_phones = int(masked_results.map(lambda result: result[2]).sum())

    remaining_emails = int(
        tickets["text"].map(lambda value: len(EMAIL_PATTERN.findall(value))).sum()
    )
    remaining_phones = int(
        tickets["text"].map(lambda value: len(PHONE_PATTERN.findall(value))).sum()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(OUTPUT_FILE, index=False)

    report = {
        "input_file": str(input_file),
        "output_file": str(OUTPUT_FILE),
        "tickets_processed": int(len(tickets)),
        "email_addresses_masked": total_emails,
        "phone_numbers_masked": total_phones,
        "email_matches_remaining": remaining_emails,
        "phone_matches_remaining": remaining_phones,
        "masked_columns": ["text"],
        "preserved_identifier_columns": ["ticket_id", "customer_id", "order_id"],
        "masking_tokens": {"email": EMAIL_TOKEN, "phone": PHONE_TOKEN},
        "validation_passed": remaining_emails == 0 and remaining_phones == 0,
        "limitations": [
            "Detects email addresses and common Indian mobile-number formats.",
            "Does not identify every possible form of personal or sensitive information.",
            "Identifier columns are preserved for downstream joins and must not be sent as free text to an external service.",
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"PII-masked tickets saved to: {OUTPUT_FILE}")
    print(f"Masking report saved to: {REPORT_FILE}")
    print(f"Tickets processed: {len(tickets)}")
    print(f"Email addresses masked: {total_emails}")
    print(f"Phone numbers masked: {total_phones}")
    print(f"Remaining email matches: {remaining_emails}")
    print(f"Remaining phone matches: {remaining_phones}")
    print(f"Masking validation passed: {report['validation_passed']}")

    return tickets


if __name__ == "__main__":
    mask_ticket_file()
