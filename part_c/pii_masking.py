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

# Matches common 10-digit phone numbers, including optional +91/91/0 prefix
# and common spaces, dots or hyphens. Boundaries avoid masking digits that
# are part of a longer alphanumeric identifier.
PHONE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:\+?91[\s.-]?)?"
    r"(?:0[\s.-]?)?"
    r"[6-9]\d{2}[\s.-]?\d{3}[\s.-]?\d{4}"
    r"(?![A-Za-z0-9])"
)


def mask_pii(value):
    """
    Return (masked_text, email_count, phone_count).

    Email masking runs first. Phone masking is limited to common Indian
    mobile-number formats to reduce accidental masking of order numbers,
    dates and other numeric values.
    """
    if pd.isna(value):
        return "", 0, 0

    text = str(value)

    text, email_count = EMAIL_PATTERN.subn("[EMAIL_REDACTED]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE_REDACTED]", text)

    return text, email_count, phone_count


def mask_ticket_file(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Mask PII in the ticket text column and save the masked dataset."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Cleaned ticket file not found: {input_file}. "
            "Run part_c/ticket_cleaning.py first."
        )

    tickets = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    if "text" not in tickets.columns:
        raise ValueError("Input file must contain a 'text' column.")

    masked_results = tickets["text"].map(mask_pii)

    tickets["text"] = masked_results.map(lambda result: result[0])

    total_emails_masked = int(masked_results.map(lambda result: result[1]).sum())
    total_phones_masked = int(masked_results.map(lambda result: result[2]).sum())

    # A simple post-mask scan provides a useful quality check.
    remaining_emails = int(
        tickets["text"].map(lambda value: len(EMAIL_PATTERN.findall(value))).sum()
    )
    remaining_phones = int(
        tickets["text"].map(lambda value: len(PHONE_PATTERN.findall(value))).sum()
    )

    report = {
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "tickets_processed": int(len(tickets)),
        "email_addresses_masked": total_emails_masked,
        "phone_numbers_masked": total_phones_masked,
        "email_matches_remaining": remaining_emails,
        "phone_matches_remaining": remaining_phones,
        "masked_columns": ["text"],
        "unchanged_identifier_columns": [
            column
            for column in ("ticket_id", "customer_id", "order_id")
            if column in tickets.columns
        ],
        "masking_tokens": {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tickets.to_csv(OUTPUT_FILE, index=False)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"PII-masked tickets saved to: {OUTPUT_FILE}")
    print(f"Masking report saved to: {REPORT_FILE}")
    print(f"Email addresses masked: {total_emails_masked}")
    print(f"Phone numbers masked: {total_phones_masked}")
    print(f"Remaining email matches: {remaining_emails}")
    print(f"Remaining phone matches: {remaining_phones}")

    return tickets


if __name__ == "__main__":
    mask_ticket_file()
