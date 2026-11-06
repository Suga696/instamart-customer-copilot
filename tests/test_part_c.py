"""Unit tests for Part C ticket cleaning, classification and routing.

Run from the repository root:
    python -m unittest tests.test_part_c -v

These tests cover the modules currently in the project. PII masking should
also be tested once part_c/pii_masking.py is added.
"""

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from part_c.ticket_cleaning import clean_ticket_text, normalize_category
from part_c.classification import classify_one, parse_model_response
from part_c.routing import route_category, route_ticket


class TestPartCTicketCleaning(unittest.TestCase):
    def test_whitespace_is_normalized(self):
        self.assertEqual(clean_ticket_text("  rider\tis\nlate  "), "rider is late")

    def test_missing_description_becomes_empty_string(self):
        self.assertEqual(clean_ticket_text(None), "")

    def test_category_is_lowercase(self):
        self.assertEqual(normalize_category("  DELIVERY "), "delivery")


class TestPartCClassification(unittest.TestCase):
    def setUp(self):
        mock_llm.reset()

    def test_empty_ticket_is_unclassified_without_model_call(self):
        result = classify_one("")
        self.assertEqual(result["predicted_category"], "unclassified")
        self.assertEqual(result["classification_status"], "empty_input")
        self.assertEqual(result["attempts"], 0)

    def test_valid_ticket_gets_allowed_category(self):
        result = classify_one("My order delivery is late and rider has not arrived.")
        self.assertIn(
            result["predicted_category"],
            {"billing", "delivery", "technical", "fraud"},
        )
        self.assertEqual(result["classification_status"], "classified")

    def test_invalid_json_response_is_rejected(self):
        class FakeMessage:
            content = "not-json"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        with self.assertRaises(ValueError):
            parse_model_response(FakeResponse())


class TestPartCRouting(unittest.TestCase):
    def test_known_categories_map_to_support_queues(self):
        self.assertEqual(route_category("billing"), "Billing Support")
        self.assertEqual(route_category("delivery"), "Delivery Support")
        self.assertEqual(route_category("technical"), "Technical Support")
        self.assertEqual(route_category("fraud"), "Security Team")

    def test_unknown_and_review_categories_go_to_manual_review(self):
        self.assertEqual(route_category("needs_review"), "Manual Review")
        self.assertEqual(route_category("unclassified"), "Manual Review")
        self.assertEqual(route_category("shipping"), "Manual Review")

    def test_fraud_is_high_priority_and_two_hour_sla(self):
        result = route_ticket({"predicted_category": "fraud", "cluster": "0"})
        self.assertEqual(result["assigned_queue"], "Security Team")
        self.assertEqual(result["priority"], "HIGH")
        self.assertEqual(result["response_sla"], "2 hours")

    def test_unclassified_ticket_requires_manual_review(self):
        result = route_ticket({"predicted_category": "unclassified", "cluster": "0"})
        self.assertEqual(result["assigned_queue"], "Manual Review")
        self.assertEqual(result["priority"], "REVIEW")


if __name__ == "__main__":
    unittest.main()
