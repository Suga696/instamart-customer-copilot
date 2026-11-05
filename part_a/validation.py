import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "validation_report.json"

EXPECTED_COLUMNS = {
    "customers_clean.csv": [
        "customer_id", "city", "signup_date", "age", "gender"
    ],
    "products_clean.csv": [
        "product_id", "product_name", "category", "price_inr"
    ],
    "orders_clean.csv": [
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes"
    ],
}


def load_clean_csv(filename):
    """Load a cleaned CSV and verify that it exists and has expected columns."""
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned file: {path}\n"
            "Run 'python part_a/cleaning.py' first."
        )

    dataframe = pd.read_csv(path)
    expected = EXPECTED_COLUMNS[filename]
    missing_columns = [col for col in expected if col not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            f"{filename} is missing required columns: {missing_columns}"
        )

    return dataframe


def add_check(checks, name, passed, details):
    """Append a consistent validation result."""
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    })


def validate_customers(customers, checks):
    """Validate customer identifiers and standardized customer attributes."""
    duplicate_ids = int(customers["customer_id"].duplicated().sum())
    blank_ids = int(
        customers["customer_id"].isna().sum()
        + customers["customer_id"].astype("string").str.strip().eq("").sum()
    )
    add_check(
        checks,
        "Customer IDs are present and unique",
        duplicate_ids == 0 and blank_ids == 0,
        {
            "duplicate_customer_ids": duplicate_ids,
            "blank_or_missing_customer_ids": blank_ids,
        },
    )

    allowed_genders = {"Male", "Female"}
    unexpected_genders = sorted(
        set(customers["gender"].dropna().astype(str)) - allowed_genders
    )
    add_check(
        checks,
        "Customer gender values are standardized",
        len(unexpected_genders) == 0,
        {"unexpected_values": unexpected_genders},
    )

    invalid_ages = customers["age"].notna() & ~customers["age"].between(18, 100)
    invalid_age_count = int(invalid_ages.sum())
    add_check(
        checks,
        "Non-missing customer ages are within 18-100",
        invalid_age_count == 0,
        {"invalid_age_count": invalid_age_count},
    )

    invalid_dates = int(
        pd.to_datetime(customers["signup_date"], errors="coerce").isna().sum()
    )
    add_check(
        checks,
        "Customer signup dates are valid",
        invalid_dates == 0,
        {"invalid_signup_dates": invalid_dates},
    )


def validate_products(products, checks):
    """Validate product identifiers and prices."""
    duplicate_ids = int(products["product_id"].duplicated().sum())
    blank_ids = int(
        products["product_id"].isna().sum()
        + products["product_id"].astype("string").str.strip().eq("").sum()
    )
    add_check(
        checks,
        "Product IDs are present and unique",
        duplicate_ids == 0 and blank_ids == 0,
        {
            "duplicate_product_ids": duplicate_ids,
            "blank_or_missing_product_ids": blank_ids,
        },
    )

    invalid_prices = products["price_inr"].notna() & (products["price_inr"] <= 0)
    invalid_price_count = int(invalid_prices.sum())
    add_check(
        checks,
        "Non-missing product prices are positive",
        invalid_price_count == 0,
        {"invalid_price_count": invalid_price_count},
    )


def validate_orders(orders, customers, products, checks):
    """Validate order IDs, business values and master-data references."""
    duplicate_ids = int(orders["order_id"].duplicated().sum())
    blank_ids = int(
        orders["order_id"].isna().sum()
        + orders["order_id"].astype("string").str.strip().eq("").sum()
    )
    add_check(
        checks,
        "Order IDs are present and unique",
        duplicate_ids == 0 and blank_ids == 0,
        {
            "duplicate_order_ids": duplicate_ids,
            "blank_or_missing_order_ids": blank_ids,
        },
    )

    allowed_statuses = {"completed", "cancelled"}
    unexpected_statuses = sorted(
        set(orders["status"].dropna().astype(str)) - allowed_statuses
    )
    add_check(
        checks,
        "Order statuses are recognized",
        len(unexpected_statuses) == 0,
        {"unexpected_statuses": unexpected_statuses},
    )

    invalid_quantity = orders["quantity"].notna() & (orders["quantity"] <= 0)
    invalid_discount = (
        orders["discount_pct"].notna()
        & ~orders["discount_pct"].between(0, 100)
    )
    invalid_amount = (
        orders["order_amount"].notna() & (orders["order_amount"] <= 0)
    )
    invalid_delivery = (
        orders["delivery_minutes"].notna()
        & (orders["delivery_minutes"] < 0)
    )

    add_check(
        checks,
        "Order quantities are positive when present",
        int(invalid_quantity.sum()) == 0,
        {"invalid_quantity_count": int(invalid_quantity.sum())},
    )
    add_check(
        checks,
        "Discount percentages are between 0 and 100 when present",
        int(invalid_discount.sum()) == 0,
        {"invalid_discount_count": int(invalid_discount.sum())},
    )
    add_check(
        checks,
        "Order amounts are positive when present",
        int(invalid_amount.sum()) == 0,
        {"invalid_amount_count": int(invalid_amount.sum())},
    )
    add_check(
        checks,
        "Delivery durations are non-negative when present",
        int(invalid_delivery.sum()) == 0,
        {"invalid_delivery_count": int(invalid_delivery.sum())},
    )

    invalid_timestamps = int(
        pd.to_datetime(orders["order_ts"], errors="coerce").isna().sum()
    )
    add_check(
        checks,
        "Order timestamps are valid",
        invalid_timestamps == 0,
        {"invalid_order_timestamps": invalid_timestamps},
    )

    customer_ids = set(customers["customer_id"].dropna().astype(str))
    product_ids = set(products["product_id"].dropna().astype(str))

    unknown_customers = (
        orders["customer_id"].isna()
        | ~orders["customer_id"].astype(str).isin(customer_ids)
    )
    unknown_products = (
        orders["product_id"].isna()
        | ~orders["product_id"].astype(str).isin(product_ids)
    )

    add_check(
        checks,
        "Order customer IDs exist in customers dataset",
        int(unknown_customers.sum()) == 0,
        {"orders_with_unknown_customer_id": int(unknown_customers.sum())},
    )
    add_check(
        checks,
        "Order product IDs exist in products dataset",
        int(unknown_products.sum()) == 0,
        {"orders_with_unknown_product_id": int(unknown_products.sum())},
    )


def build_dataset_summary(dataframes):
    """Summarize row counts and remaining missing values."""
    summary = {}

    for filename, dataframe in dataframes.items():
        summary[filename] = {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "missing_values_by_column": {
                column: int(count)
                for column, count in dataframe.isna().sum().items()
            },
            "total_missing_values": int(dataframe.isna().sum().sum()),
        }

    return summary


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART A: DATA VALIDATION")
    print("=" * 65)

    customers = load_clean_csv("customers_clean.csv")
    products = load_clean_csv("products_clean.csv")
    orders = load_clean_csv("orders_clean.csv")

    checks = []
    validate_customers(customers, checks)
    validate_products(products, checks)
    validate_orders(orders, customers, products, checks)

    passed_count = sum(item["status"] == "PASS" for item in checks)
    failed_count = sum(item["status"] == "FAIL" for item in checks)

    report = {
        "stage": "Part A - Cleaned data validation",
        "overall_status": "PASS" if failed_count == 0 else "FAIL",
        "checks_passed": passed_count,
        "checks_failed": failed_count,
        "checks": checks,
        "dataset_summary": build_dataset_summary({
            "customers_clean.csv": customers,
            "products_clean.csv": products,
            "orders_clean.csv": orders,
        }),
        "note": (
            "Missing values are reported for review. They are not treated "
            "as automatic failures because the cleaning stage intentionally "
            "preserves some missing or unparseable source values."
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nValidation checks")
    print("-" * 65)
    for item in checks:
        print(f"[{item['status']}] {item['check']}: {item['details']}")

    print("\n" + "-" * 65)
    print(f"Overall status: {report['overall_status']}")
    print(f"Checks passed:  {passed_count}")
    print(f"Checks failed:  {failed_count}")
    print(f"Report saved:   {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 65)

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
