import sqlite3
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part_a import database


class TestPartADatabase(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("PRAGMA foreign_keys = ON")
        database.create_schema(self.connection)

    def tearDown(self):
        self.connection.close()

    def test_expected_tables_and_analytical_view_exist(self):
        names = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        self.assertTrue({"customers", "products", "orders", "v_orders_detail"}.issubset(names))

    def test_insert_and_join_customer_product_order(self):
        customer_count = database.insert_customers(self.connection, [{
            "customer_id": "C_TEST", "city": "Chennai",
            "signup_date": "2025-01-01", "age": "30", "gender": "Female",
        }])
        product_count = database.insert_products(self.connection, [{
            "product_id": "P_TEST", "product_name": "Test Product",
            "category": "Grocery", "price_inr": "100",
        }])
        order_count = database.insert_orders(self.connection, [{
            "order_id": "ORD-TEST01", "customer_id": "C_TEST",
            "product_id": "P_TEST", "quantity": "1",
            "order_ts": "2025-04-01 10:00:00", "status": "completed",
            "discount_pct": "0", "order_amount": "100",
            "delivery_minutes": "35",
        }])

        self.assertEqual((customer_count, product_count, order_count), (1, 1, 1))
        result = self.connection.execute(
            "SELECT order_id, city, product_name FROM v_orders_detail"
        ).fetchone()
        self.assertEqual(result, ("ORD-TEST01", "Chennai", "Test Product"))

    def test_foreign_key_violation_is_rejected(self):
        database.insert_customers(self.connection, [{
            "customer_id": "C_TEST", "city": "Chennai",
            "signup_date": "2025-01-01", "age": "30", "gender": "Female",
        }])
        database.insert_products(self.connection, [{
            "product_id": "P_TEST", "product_name": "Test Product",
            "category": "Grocery", "price_inr": "100",
        }])
        with self.assertRaises(sqlite3.IntegrityError):
            database.insert_orders(self.connection, [{
                "order_id": "ORD-BAD001", "customer_id": "UNKNOWN",
                "product_id": "P_TEST", "quantity": "1",
                "order_ts": "2025-04-01 10:00:00", "status": "completed",
                "discount_pct": "0", "order_amount": "100",
                "delivery_minutes": "35",
            }])

    def test_database_summary_reports_counts_and_integrity(self):
        database.insert_customers(self.connection, [{
            "customer_id": "C_TEST", "city": "Chennai",
            "signup_date": "2025-01-01", "age": "30", "gender": "Female",
        }])
        database.insert_products(self.connection, [{
            "product_id": "P_TEST", "product_name": "Test Product",
            "category": "Grocery", "price_inr": "100",
        }])
        database.insert_orders(self.connection, [{
            "order_id": "ORD-TEST01", "customer_id": "C_TEST",
            "product_id": "P_TEST", "quantity": "1",
            "order_ts": "2025-04-01 10:00:00", "status": "completed",
            "discount_pct": "0", "order_amount": "100",
            "delivery_minutes": "35",
        }])
        summary = database.get_database_summary(self.connection)
        self.assertEqual(summary["table_row_counts"]["orders"], 1)
        self.assertEqual(summary["joined_orders_in_view"], 1)
        self.assertEqual(summary["foreign_key_errors"], 0)


if __name__ == "__main__":
    unittest.main()
