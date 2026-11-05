"""
Part C - Support ticket classification
--------------------------------------
Classifies PII-masked ticket descriptions with the supplied offline mock_llm.py.

Input:
    data/processed/part_c/pii_masked_tickets.csv

Outputs:
    data/processed/part_c/classified_tickets.csv
    data/processed/part_c/classification_report.json

The supplied mock_llm.py is imported from the repository root and is never
modified. Empty descriptions are retained with an explicit "unclassified"
result. Long descriptions are sent without truncation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm  # supplied project file; do not edit

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "part_c" / "pii_masked_tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_c"
OUTPUT_FILE = OUTPUT_DIR / "classified_tickets.csv"
REPORT_FILE = OUTPUT_DIR / "classification_report.json"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """CLASSIFY

You are a support-ticket classification assistant.
Classify the customer's issue into exactly one of these categories:
billing, delivery, technical, fraud.

Return only valid JSON in this exact structure:
{"category": "one_allowed_category", "reason": "short explanation"}

Do not return markdown, code fences, or additional text.
Use the examples below to understand the category boundaries.

EXAMPLES:
Ticket: I was charged twice and need the extra payment refunded.
Output: {"category": "billing", "reason": "duplicate payment"}

Ticket: My order is late and the rider has not arrived.
Output: {"category": "delivery", "reason": "late delivery"}

Ticket: The app crashes when I open the cart.
Output: {"category": "technical", "reason": "application error"}

Ticket: There is an unauthorized transaction on my card.
Output: {"category": "fraud", "reason": "unauthorized transaction"}
"""


def parse_model_response(response):
    """Validate the mock model response and return its category and reason."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Model response did not contain message content.") from exc

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned empty content.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Model JSON must be an object.")

    category = payload.get("category")
    reason = payload.get("reason", "")

    if not isinstance(category, str):
        raise ValueError("Model JSON is missing a string category.")

    category = category.strip().lower()
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported category returned: {category!r}")

    if not isinstance(reason, str):
        reason = str(reason)

    return category, reason.strip()


def classify_one(ticket_text, max_attempts=MAX_ATTEMPTS):
    """
    Classify one masked ticket.

    Transient mock-model failures and invalid model responses are retried.
    If all attempts fail, the ticket is marked for manual review rather than
    assigning an unsupported or guessed category.
    """
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        return {
            "predicted_category": "unclassified",
            "classification_reason": "Empty ticket description",
            "classification_status": "empty_input",
            "attempts": 0,
            "last_error": "",
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket_text},
    ]

    last_error = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = mock_llm.chat(messages)
            category, reason = parse_model_response(response)

            return {
                "predicted_category": category,
                "classification_reason": reason,
                "classification_status": "classified",
                "attempts": attempt,
                "last_error": "",
            }

        except mock_llm.TransientLLMError as exc:
            last_error = f"TransientLLMError: {exc}"
        except (ValueError, TypeError, AttributeError, IndexError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "predicted_category": "needs_review",
        "classification_reason": "Automatic classification did not produce a valid result",
        "classification_status": "fallback_after_retries",
        "attempts": max_attempts,
        "last_error": last_error,
    }


def classify_tickets(input_file: Path = INPUT_FILE) -> pd.DataFrame:
    """Classify all masked tickets and save predictions plus a run report."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Masked ticket file not found: {input_file}. "
            "Run ticket_cleaning.py and pii_masking.py first."
        )

    tickets = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    required_columns = {"ticket_id", "customer_id", "text"}
    missing_columns = required_columns - set(tickets.columns)
    if missing_columns:
        raise ValueError(
            "Input file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Reset deterministic mock-model counters once for this complete batch.
    mock_llm.reset()

    results = []
    for ticket_text in tickets["text"]:
        results.append(classify_one(ticket_text))

    result_frame = pd.DataFrame(results)
    classified = pd.concat(
        [tickets.reset_index(drop=True), result_frame.reset_index(drop=True)],
        axis=1,
    )

    # Build a compact, reproducible run report.
    category_counts = {
        str(category): int(count)
        for category, count in classified["predicted_category"].value_counts().items()
    }
    status_counts = {
        str(status): int(count)
        for status, count in classified["classification_status"].value_counts().items()
    }

    report = {
        "input_file": str(input_file.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "tickets_processed": int(len(classified)),
        "category_counts": category_counts,
        "classification_status_counts": status_counts,
        "total_model_attempts": int(classified["attempts"].sum()),
        "maximum_attempts_per_ticket": MAX_ATTEMPTS,
        "empty_descriptions": int(
            (classified["classification_status"] == "empty_input").sum()
        ),
        "manual_review_fallbacks": int(
            (classified["classification_status"] == "fallback_after_retries").sum()
        ),
        "pii_masked_input_used": True,
        "mock_llm_reset_before_batch": True,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classified.to_csv(OUTPUT_FILE, index=False)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Classified tickets saved to: {OUTPUT_FILE}")
    print(f"Classification report saved to: {REPORT_FILE}")
    print(f"Tickets processed: {len(classified)}")
    print(f"Category counts: {category_counts}")
    print(f"Status counts: {status_counts}")

    return classified


if __name__ == "__main__":
    classify_tickets()
