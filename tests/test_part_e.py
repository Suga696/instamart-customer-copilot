"""Unit tests for the five Instamart Part E policy modules.

Run from the repository root with:
    python -m unittest discover -s tests -p 'test_part_e.py' -v

Only Python's standard-library unittest/tempfile/pathlib modules and the
project's already-permitted pandas dependency are used.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from part_e import (
    retention_policy,
    fraud_prevention_policy,
    delivery_improvement_policy,
    customer_support_policy,
    business_recommendations,
)


class PartEPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.proc = self.root / "data" / "processed"
        self.part_b = self.proc / "part_b"
        self.part_c = self.proc / "part_c"
        self.part_d = self.proc / "part_d"
        self.out = self.proc / "part_e"
        for folder in (self.part_b, self.part_c, self.part_d, self.out):
            folder.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_retention_policy_generates_csv_and_json(self):
        pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "cluster": [0, 0, 1],
        }).to_csv(self.part_b / "customer_segments.csv", index=False)
        pd.DataFrame({
            "cluster": [0, 1],
            "total_order_amount": [1500.0, 300.0],
            "total_orders": [5.0, 1.0],
            "average_discount_pct": [5.0, 10.0],
            "days_since_last_order": [4.0, 45.0],
        }).to_csv(self.part_b / "segment_profile.csv", index=False)

        with patch.object(retention_policy, "PROC", self.proc), patch.object(retention_policy, "OUT", self.out):
            retention_policy.main()

        result = pd.read_csv(self.out / "retention_recommendations.csv")
        report = json.loads((self.out / "retention_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(result), 2)
        self.assertEqual(int(result.loc[result.cluster_id == 0, "customers"].iloc[0]), 2)
        self.assertIn("caveat", report)

    def test_fraud_prevention_policy_preserves_human_review_safeguards(self):
        pd.DataFrame({"review_priority": ["LOW", "MEDIUM", "HIGH", "HIGH"]}).to_csv(
            self.part_c / "fraud_anomaly_results.csv", index=False
        )
        with patch.object(fraud_prevention_policy, "ROOT", self.root), patch.object(fraud_prevention_policy, "OUT", self.out):
            fraud_prevention_policy.main()
        report = json.loads((self.out / "fraud_prevention_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(report["screening_counts"]["HIGH"], 2)
        self.assertTrue(any("not proof of fraud" in item for item in report["safeguards"]))
        self.assertTrue((self.out / "fraud_review_policy_counts.csv").exists())

    def test_delivery_improvement_policy_generates_city_and_hour_recommendations(self):
        pd.DataFrame({
            "city": ["Chennai", "Bengaluru"],
            "delay_rate_pct": [35.0, 12.0],
        }).to_csv(self.part_d / "delivery_city_analysis.csv", index=False)
        pd.DataFrame({
            "hour": [12, 20, 21, 22],
            "avg_delivery_minutes": [28.0, 42.0, 39.0, 35.0],
        }).to_csv(self.part_d / "delivery_hourly_analysis.csv", index=False)
        with patch.object(delivery_improvement_policy, "PROC", self.proc), patch.object(delivery_improvement_policy, "OUT", self.out):
            delivery_improvement_policy.main()
        report = json.loads((self.out / "delivery_improvement_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["highest_average_delivery_hours"]), 3)
        city = pd.read_csv(self.out / "delivery_city_recommendations.csv")
        self.assertIn("recommendation", city.columns)
        self.assertIn("limitations", report)

    def test_customer_support_policy_generates_queue_summary_and_limitation(self):
        pd.DataFrame({
            "assigned_queue": ["Billing Team", "Security Team", "Billing Team"],
            "priority": ["Medium", "High", "Low"],
        }).to_csv(self.part_c / "routed_tickets.csv", index=False)
        with patch.object(customer_support_policy, "ROOT", self.root), patch.object(customer_support_policy, "OUT", self.out):
            customer_support_policy.main()
        report = json.loads((self.out / "customer_support_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["routing_summary"]), 3)
        self.assertIn("Premium mapping", report["limitation"])
        self.assertTrue((self.out / "support_queue_summary.csv").exists())

    def test_business_recommendations_consolidates_four_policy_reports(self):
        inputs = {
            "retention_policy.json": {"policy": []},
            "fraud_prevention_policy.json": {"safeguards": []},
            "delivery_improvement_policy.json": {"limitations": []},
            "customer_support_policy.json": {"rules": []},
        }
        for filename, payload in inputs.items():
            (self.out / filename).write_text(json.dumps(payload), encoding="utf-8")
        with patch.object(business_recommendations, "BASE", self.out):
            business_recommendations.main()
        report = json.loads((self.out / "business_policy_report.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["recommendations"]), 4)
        self.assertIn("governance", report)

    def test_business_recommendations_fails_clearly_when_input_missing(self):
        with patch.object(business_recommendations, "BASE", self.out):
            with self.assertRaisesRegex(FileNotFoundError, "Run the corresponding Part E script first"):
                business_recommendations.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
