"""
Part A - Data cleaning for the Instamart capstone.

Run from the repository root:
    python part_a/cleaning.py

Inputs:
    data/raw/customers.csv
    data/raw/products.csv
    data/raw/orders.csv

Outputs:
    data/processed/customers_clean.csv
    data/processed/products_clean.csv
    data/processed/orders_clean.csv
    data/processed/cleaning_report.json

The raw source CSV files are never overwritten.

Cleaning decisions:
- Standardize customer city and gender values.
- Convert impossible customer ages (outside 18-100) to missing values.
- Convert numeric fields safely; unparseable values become missing.
- Remove exact duplicate order rows.
- For repeated order IDs, prefer a completed/cancelled record over an
  earlier pending record. If records have the same status priority, keep
  the record with the latest order timestamp.
- Convert negative delivery durations and out-of-range discounts to missing.
- Preserve rows with missing values so that data loss is visible and can be
  handled explicitly in later analysis.
"""

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def read_raw_csv(filename):
    """Read a raw CSV and fail with a helpful message if it is missing."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run 'python generate_data.py' "
            "from the repository root first."
        )
    return pd.read_csv(path)


def clean_customers(customers):
    """Standardize customer attributes and handle invalid ages."""
    df = customers.copy()
    original_rows = len(df)

    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df["city"] = (
        df["city"].astype("string").str.strip().str.title()
    )

    gender_map = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
    }
    df["gender"] = (
        df["gender"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(gender_map)
    )

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    invalid_age_mask = df["age"].notna() & ~df["age"].between(18, 100)
    invalid_age_count = int(invalid_age_mask.sum())
    df.loc[invalid_age_mask, "age"] = pd.NA
    df["age"] = df["age"].astype("Int64")

    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    duplicate_count = int(df.duplicated(subset=["customer_id"]).sum())
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

    report = {
        "rows_before": int(original_rows),
        "rows_after": int(len(df)),
        "duplicate_customer_ids_removed": duplicate_count,
        "invalid_ages_set_missing": invalid_age_count,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
    }
    return df, report


def clean_products(products):
    """Standardize product text and validate product prices."""
    df = products.copy()
    original_rows = len(df)

    for column in ["product_id", "product_name", "category"]:
        df[column] = df[column].astype("string").str.strip()

    df["category"] = df["category"].str.title()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")

    invalid_price_mask = df["price_inr"].notna() & (df["price_inr"] <= 0)
    invalid_price_count = int(invalid_price_mask.sum())
    df.loc[invalid_price_mask, "price_inr"] = pd.NA

    duplicate_count = int(df.duplicated(subset=["product_id"]).sum())
    df = df.drop_duplicates(subset=["product_id"], keep="first")

    report = {
        "rows_before": int(original_rows),
        "rows_after": int(len(df)),
        "duplicate_product_ids_removed": duplicate_count,
        "invalid_prices_set_missing": invalid_price_count,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
    }
    return df, report


def clean_orders(orders):
    """Clean order values and resolve exact and partial duplicate records."""
    df = orders.copy()
    original_rows = len(df)

    # Normalize text fields before duplicate handling.
    for column in ["order_id", "customer_id", "product_id", "status"]:
        df[column] = df[column].astype("string").str.strip()

    df["status"] = df["status"].str.lower()
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")

    # Convert numeric columns without crashing on values such as N/A or error.
    numeric_columns = [
        "quantity", "discount_pct", "order_amount", "delivery_minutes"
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates(keep="first").copy()

    # Validate numeric business fields.
    invalid_quantity_mask = df["quantity"].notna() & (df["quantity"] <= 0)
    invalid_discount_mask = (
        df["discount_pct"].notna()
        & ~df["discount_pct"].between(0, 100)
    )
    invalid_amount_mask = (
        df["order_amount"].notna() & (df["order_amount"] <= 0)
    )
    negative_delivery_mask = (
        df["delivery_minutes"].notna() & (df["delivery_minutes"] < 0)
    )

    invalid_quantity_count = int(invalid_quantity_mask.sum())
    invalid_discount_count = int(invalid_discount_mask.sum())
    invalid_amount_count = int(invalid_amount_mask.sum())
    negative_delivery_count = int(negative_delivery_mask.sum())

    df.loc[invalid_quantity_mask, "quantity"] = pd.NA
    df.loc[invalid_discount_mask, "discount_pct"] = pd.NA
    df.loc[invalid_amount_mask, "order_amount"] = pd.NA
    df.loc[negative_delivery_mask, "delivery_minutes"] = pd.NA

    # The generator creates an earlier pending row for some orders.
    # Prefer the final business status over pending; within the same
    # priority, prefer the latest timestamp.
    status_priority = {
        "completed": 2,
        "cancelled": 2,
        "pending": 1,
    }
    df["_status_priority"] = df["status"].map(status_priority).fillna(0)
    df = df.sort_values(
        by=["_status_priority", "order_ts"],
        ascending=[False, False],
        na_position="last",
        kind="stable",
    )

    duplicate_order_ids = int(df.duplicated(subset=["order_id"]).sum())
    df = df.drop_duplicates(subset=["order_id"], keep="first").copy()
    df = df.drop(columns=["_status_priority"])

    # Store timestamps consistently for CSV output.
    df["order_ts"] = df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Keep quantities and discounts numeric, while preserving missing values.
    df = df.sort_values("order_id", kind="stable").reset_index(drop=True)

    report = {
        "rows_before": int(original_rows),
        "exact_duplicate_rows_removed": exact_duplicates,
        "duplicate_order_records_removed_after_status_resolution": duplicate_order_ids,
        "rows_after": int(len(df)),
        "invalid_quantities_set_missing": invalid_quantity_count,
        "invalid_discounts_set_missing": invalid_discount_count,
        "invalid_amounts_set_missing": invalid_amount_count,
        "negative_delivery_times_set_missing": negative_delivery_count,
        "missing_values_after_cleaning": int(df.isna().sum().sum()),
    }
    return df, report


def validate_foreign_keys(customers, products, orders):
    """Count order rows whose customer or product IDs are not in master data."""
    customer_ids = set(customers["customer_id"].dropna().astype(str))
    product_ids = set(products["product_id"].dropna().astype(str))

    missing_customer = ~orders["customer_id"].astype(str).isin(customer_ids)
    missing_product = ~orders["product_id"].astype(str).isin(product_ids)

    return {
        "orders_with_unknown_customer_id": int(missing_customer.sum()),
        "orders_with_unknown_product_id": int(missing_product.sum()),
    }


def main():
    print("=" * 65)
    print("INSTAMART CAPSTONE - PART A: DATA CLEANING")
    print("=" * 65)

    customers_raw = read_raw_csv("customers.csv")
    products_raw = read_raw_csv("products.csv")
    orders_raw = read_raw_csv("orders.csv")

    customers_clean, customers_report = clean_customers(customers_raw)
    products_clean, products_report = clean_products(products_raw)
    orders_clean, orders_report = clean_orders(orders_raw)

    foreign_key_report = validate_foreign_keys(
        customers_clean, products_clean, orders_clean
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    customers_clean.to_csv(
        PROCESSED_DIR / "customers_clean.csv", index=False
    )
    products_clean.to_csv(
        PROCESSED_DIR / "products_clean.csv", index=False
    )
    orders_clean.to_csv(
        PROCESSED_DIR / "orders_clean.csv", index=False
    )

    report = {
        "stage": "Part A - Data cleaning",
        "cleaning_rules": {
            "city": "Trim whitespace and standardize to title case",
            "gender": "Map male/m and female/f variants to standard labels",
            "age": "Values outside 18-100 are set to missing",
            "orders": (
                "Remove exact duplicates; prefer completed/cancelled rows "
                "over pending rows for repeated order IDs"
            ),
            "invalid_numeric_values": (
                "Unparseable or invalid values are set to missing; rows "
                "are retained for transparent downstream handling"
            ),
        },
        "customers": customers_report,
        "products": products_report,
        "orders": orders_report,
        "foreign_key_checks": foreign_key_report,
        "output_files": [
            "data/processed/customers_clean.csv",
            "data/processed/products_clean.csv",
            "data/processed/orders_clean.csv",
        ],
    }

    report_path = PROCESSED_DIR / "cleaning_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nCleaning summary")
    print("-" * 45)
    for dataset_name, dataset_report in [
        ("Customers", customers_report),
        ("Products", products_report),
        ("Orders", orders_report),
    ]:
        print(f"\n{dataset_name}:")
        for key, value in dataset_report.items():
            print(f"  {key}: {value}")

    print("\nForeign-key checks:")
    for key, value in foreign_key_report.items():
        print(f"  {key}: {value}")

    print("\nCLEANING COMPLETED")
    print(f"Cleaned files saved in: {PROCESSED_DIR}")
    print(f"Report saved to: {report_path}")
    print("Raw files in data/raw were not modified.")


if __name__ == "__main__":
    main()
