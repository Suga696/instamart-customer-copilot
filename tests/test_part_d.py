"""Unit tests for Part D policy retrieval and refund-agent tools.

Run from the repository root:
    python -m unittest tests.test_part_d -v
"""

import sqlite3
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part_d.policy_retrieval import retrieve_documents, ABSTENTION
from part_d.refund_agent import (
    get_order,
    execute_tool,
    outcome_from_response,
)


class TestPartDPolicyRetrieval(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "refund_policy.txt": (
                "Completed orders delivered in more than 45 minutes receive "
                "a full refund. Refunds above INR 1000 require human approval."
            ),
            "payment_policy.txt": (
                "Failed payments where money was deducted are returned within 48 hours."
            ),
            "privacy_policy.txt": (
                "Customer email and phone details are masked before automated processing."
            ),
        }

    def test_refund_question_retrieves_refund_policy(self):
        ranked, score = retrieve_documents(
            "What delivery time makes an order eligible for a full refund?",
            self.documents,
        )
        self.assertTrue(ranked)
        self.assertEqual(ranked[0][0], "refund_policy.txt")
        self.assertGreater(score, 0)

    def test_unrelated_question_has_no_document_match(self):
        ranked, score = retrieve_documents(
            "Who won the cricket world cup final?",
            self.documents,
        )
        self.assertEqual(ranked, [])
        self.assertEqual(score, 0.0)

    def test_abstention_text_is_explicit(self):
        self.assertIn("could not find", ABSTENTION.lower())


class TestPartDRefundTools(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT,
                status TEXT,
                order_amount REAL,
                delivery_minutes INTEGER
            )
            """
        )
        self.connection.execute(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            ("ORD-123456", "C0001", "completed", 750.0, 52),
        )
        self.connection.commit()
        self.policy = {
            "refund_policy.txt": (
                "Completed orders delivered in more than 45 minutes "
                "are eligible for a full refund."
            )
        }
        self.action_log = []

    def tearDown(self):
        self.connection.close()

    def test_get_existing_order(self):
        order = get_order(self.connection, "ORD-123456")
        self.assertEqual(order["status"], "completed")
        self.assertEqual(order["order_amount"], 750.0)
        self.assertEqual(order["delivery_minutes"], 52)

    def test_unknown_order_returns_not_found_data(self):
        order = get_order(self.connection, "ORD-999999")
        self.assertEqual(order["error"], "order_not_found")

    def test_refund_calculation_requires_delivery_over_45_minutes(self):
        result = execute_tool(
            "calculate_refund",
            {"order_amount": 750, "delivery_minutes": 52},
            self.connection, self.policy, self.action_log,
        )
        self.assertEqual(result["refund_inr"], 750.0)

        on_time = execute_tool(
            "calculate_refund",
            {"order_amount": 750, "delivery_minutes": 45},
            self.connection, self.policy, self.action_log,
        )
        self.assertEqual(on_time["refund_inr"], 0.0)

    def test_missing_delivery_duration_does_not_calculate_refund(self):
        result = execute_tool(
            "calculate_refund",
            {"order_amount": 750, "delivery_minutes": None},
            self.connection, self.policy, self.action_log,
        )
        self.assertIn("error", result)
        self.assertEqual(result["refund_inr"], 0)

    def test_refund_tool_is_simulated_not_real(self):
        result = execute_tool(
            "issue_refund",
            {"order_id": "ORD-123456", "refund_inr": 750},
            self.connection, self.policy, self.action_log,
        )
        self.assertEqual(result["status"], "simulated")
        self.assertIn("simulated_refund", self.action_log)

    def test_human_escalation_is_recorded(self):
        result = execute_tool(
            "escalate_to_human",
            {"order_id": "ORD-123456", "refund_inr": 1500},
            self.connection, self.policy, self.action_log,
        )
        self.assertEqual(result["status"], "escalated")
        self.assertEqual(result["assigned_to"], "human support lead")
        self.assertIn("human_escalation", self.action_log)

    def test_response_outcome_labels(self):
        self.assertEqual(outcome_from_response("REFUND: INR 500 issued."), "REFUND")
        self.assertEqual(outcome_from_response("ESCALATE: sent to lead."), "ESCALATE")
        self.assertEqual(outcome_from_response("NOT_FOUND: no such order."), "NOT_FOUND")
        self.assertEqual(outcome_from_response("unexpected response"), "FALLBACK")


if __name__ == "__main__":
    unittest.main()
