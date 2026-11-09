import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part_b.segmentation import select_numeric_features
from part_b.churn_prediction import normalize_target, select_features


class TestPartBSegmentation(unittest.TestCase):
    def test_numeric_feature_selection_excludes_identifier_and_target(self):
        frame = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "total_orders": [2, 5, 8],
            "average_order_value": [100.0, 150.0, 200.0],
            "churn": [0, 1, 0],
        })
        features, columns = select_numeric_features(frame)
        self.assertEqual(set(columns), {"total_orders", "average_order_value"})
        self.assertEqual(list(features.columns), columns)
        self.assertNotIn("churn", columns)

    def test_segmentation_rejects_data_without_numeric_behaviour_features(self):
        frame = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "segment": ["A", "B"],
        })
        with self.assertRaises(ValueError):
            select_numeric_features(frame)


class TestPartBChurn(unittest.TestCase):
    def test_normalize_binary_numeric_target(self):
        result = normalize_target(pd.Series([0, 1, 0, 1]))
        self.assertEqual(result.tolist(), [0, 1, 0, 1])

    def test_normalize_text_binary_target(self):
        result = normalize_target(pd.Series(["No", "Yes", "active", "inactive"]))
        self.assertEqual(result.tolist(), [0, 1, 0, 1])

    def test_reject_non_binary_target(self):
        with self.assertRaises(ValueError):
            normalize_target(pd.Series(["yes", "maybe", "no"]))

    def test_churn_feature_selection_excludes_customer_id_and_target(self):
        frame = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4"],
            "orders_count": [1, 2, 3, 4],
            "total_spend": [10.0, 20.0, 30.0, 40.0],
            "churn": [0, 1, 0, 1],
        })
        features, columns = select_features(frame)
        self.assertEqual(set(columns), {"orders_count", "total_spend"})
        self.assertEqual(len(features), 4)


if __name__ == "__main__":
    unittest.main()
