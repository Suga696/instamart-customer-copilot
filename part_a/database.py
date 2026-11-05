Run from the repository root, after cleaning.py and validation.py:
    python part_a/database.py

Reads cleaned CSV files from data/processed/ and creates:
    data/database/instamart.db

Tables:
    customers
    products
    orders

Also creates:
    v_orders_detail - a view joining orders with customer and product details.

The script recreates the database on each run so that results are reproducible.
"""

import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_DIR = PROJECT_ROOT / "data" / "database"
DATABASE_PATH = DATABASE_DIR / "instamart.db"


def read_csv_rows(filename):
    """Read a cleaned CSV as dictionaries."""
    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned file: {path}\n"
            "Run 'python part_a/cleaning.py' first."
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def optional_int(value):
    """Convert a CSV value to int, or return None for a blank value."""
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def optional_float(value):
    """Convert a CSV value to float, or return None for a blank value."""
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def create_schema(connection):
    """Create relational tables, constraints, indexes and an analytical view."""
    cursor = connection.cursor()

    # Drop dependent objects first to make repeated runs deterministic.
    cursor.executescript(
        """
        DROP VIEW IF EXISTS v_orders_detail;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            city TEXT,
            signup_date TEXT,
            age INTEGER CHECK (age IS NULL OR age BETWEEN 18 AND 100),
            gender TEXT CHECK (
                gender IS NULL OR gender IN ('Male', 'Female')
            )
        );

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            price_inr REAL CHECK (
                price_inr IS NULL OR price_inr > 0
            )
        );

        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_id TEXT NOT NULL,
            quantity INTEGER CHECK (
                quantity IS NULL OR quantity > 0
            ),
            order_ts TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('completed', 'cancelled')
            ),
            discount_pct REAL CHECK (
                discount_pct IS NULL OR discount_pct BETWEEN 0 AND 100
            ),
            order_amount REAL CHECK (
                order_amount IS NULL OR order_amount > 0
            ),
            delivery_minutes INTEGER CHECK (
                delivery_minutes IS NULL OR delivery_minutes >= 0
            ),
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE INDEX idx_orders_customer_id ON orders(customer_id);
        CREATE INDEX idx_orders_product_id ON orders(product_id);
        CREATE INDEX idx_orders_order_ts ON orders(order_ts);
        CREATE INDEX idx_orders_status ON orders(status);

        CREATE VIEW v_orders_detail AS
        SELECT
            o.order_id,
            o.order_ts,
            o.status,
            o.quantity,
            o.discount_pct,
            o.order_amount,
            o.delivery_minutes,
            c.customer_id,
            c.city,
            c.age,
            c.gender,
            p.product_id,
            p.product_name,
            p.category,
            p.price_inr
        FROM orders AS o
        JOIN customers AS c
            ON o.customer_id = c.customer_id
        JOIN products AS p
            ON o.product_id = p.product_id;
        """
    )


def insert_customers(connection, rows):
    """Insert cleaned customer records."""
    sql = """
        INSERT INTO customers (
            customer_id, city, signup_date, age, gender
        ) VALUES (?, ?, ?, ?, ?)
    """

    values = [
        (
            row["customer_id"],
            row.get("city") or None,
            row.get("signup_date") or None,
            optional_int(row.get("age")),
            row.get("gender") or None,
        )
        for row in rows
    ]
    connection.executemany(sql, values)
    return len(values)


def insert_products(connection, rows):
    """Insert cleaned product records."""
    sql = """
        INSERT INTO products (
            product_id, product_name, category, price_inr
        ) VALUES (?, ?, ?, ?)
    """

    values = [
        (
            row["product_id"],
            row["product_name"],
            row["category"],
            optional_float(row.get("price_inr")),
        )
        for row in rows
    ]
    connection.executemany(sql, values)
    return len(values)


def insert_orders(connection, rows):
    """Insert cleaned order records."""
    sql = """
        INSERT INTO orders (
            order_id, customer_id, product_id, quantity, order_ts,
            status, discount_pct, order_amount, delivery_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = [
        (
            row["order_id"],
            row["customer_id"],
            row["product_id"],
            optional_int(row.get("quantity")),
            row["order_ts"],
            row["status"],
            optional_float(row.get("discount_pct")),
            optional_float(row.get("order_amount")),
            optional_int(row.get("delivery_minutes")),
        )
        for row in rows
    ]
    connection.executemany(sql, values)
    return len(values)


def get_database_summary(connection):
    """Return table counts and foreign-key integrity results."""
    cursor = connection.cursor()

    counts = {}
    for table in ["customers", "products", "orders"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = int(cursor.fetchone()[0])

    cursor.execute("PRAGMA foreign_key_check")
    foreign_key_errors = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM v_orders_detail")
    joined_order_count = int(cursor.fetchone()[0])

    return {
        "table_row_counts": counts,
        "joined_orders_in_view": joined_order_count,
        "foreign_key_errors": len(foreign_key_errors),
    }


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART A: SQLITE DATABASE CREATION")
    print("=" * 65)

    customers = read_csv_rows("customers_clean.csv")
    products = read_csv_rows("products_clean.csv")
    orders = read_csv_rows("orders_clean.csv")

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    # Enable FK enforcement for this connection. The transaction is rolled
    # back if any row violates a primary key, foreign key or CHECK constraint.
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        connection.execute("PRAGMA foreign_keys = ON")

        with connection:
            create_schema(connection)
            inserted_customers = insert_customers(connection, customers)
            inserted_products = insert_products(connection, products)
            inserted_orders = insert_orders(connection, orders)

        summary = get_database_summary(connection)

        if summary["foreign_key_errors"] != 0:
            raise RuntimeError(
                f"Foreign-key validation failed: {summary['foreign_key_errors']} errors"
            )

        if summary["joined_orders_in_view"] != summary["table_row_counts"]["orders"]:
            raise RuntimeError(
                "The analytical view does not contain every order. "
                "Check customer and product relationships."
            )

    finally:
        connection.close()

    print("\nDatabase created successfully.")
    print(f"Location: {DATABASE_PATH.relative_to(PROJECT_ROOT)}")
    print("\nRows inserted:")
    print(f"  Customers: {inserted_customers}")
    print(f"  Products:  {inserted_products}")
    print(f"  Orders:    {inserted_orders}")
    print("\nDatabase checks:")
    print(f"  Orders available in joined view: {summary['joined_orders_in_view']}")
    print(f"  Foreign-key errors:              {summary['foreign_key_errors']}")
    print("\nTables: customers, products, orders")
    print("View:   v_orders_detail")
    print("=" * 65)


if __name__ == "__main__":
    main()
