# Instamart Customer Intelligence & Support Copilot

An end-to-end Python capstone using synthetic Instamart data for data
quality, customer intelligence, churn modelling, support-ticket
analytics, anomaly detection, policy retrieval, business
recommendations, and an integrated HTML dashboard.

> **Data limitation:** All data is synthetic. Churn labels are derived
> from a 30-day inactivity rule, and fraud flags identify unusual
> transactions for human review. They are not verified churn or
> confirmed fraud labels.

## Objectives

-   Ingest, clean, validate, and store customer, product, order, and
    ticket data.
-   Engineer customer features and segment customers.
-   Build and compare churn prediction models.
-   Clean, mask, classify, and route support tickets.
-   Detect unusual order patterns with anomaly detection.
-   Retrieve policy information and evaluate answerability.
-   Generate retention, fraud prevention, delivery, and support
    recommendations.
-   Consolidate project results in an HTML dashboard.

## Technology and constraints

-   Python 3.11+ and SQLite 3.25+.
-   Permitted libraries only: Python standard library, `pandas`,
    `numpy`, `scikit-learn`, and `joblib`.
-   No Streamlit or additional third-party dependencies.
-   Keep the supplied `generate_data.py` and `mock_llm.py` unchanged.
-   The supplied mock LLM runs offline; no API key is required.

Install dependencies:

``` bash
python -m pip install -r requirements.txt
```

`requirements.txt` should contain:

``` text
pandas
numpy
scikit-learn
joblib
```

## Repository structure

``` text
instamart-customer-copilot/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── policies/
│   ├── processed/
│   ├── database/
│   └── reports/
├── part_a/
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── validation.py
│   ├── database.py
│   └── data_quality_report.py
├── part_b/
│   ├── feature_engineering.py
│   ├── segmentation.py
│   ├── churn_prediction.py
│   ├── churn_target.py
│   ├── churn_model_comparison.py
│   └── customer_intelligence_report.py
├── part_c/
│   ├── ticket_cleaning.py
│   ├── pii_masking.py
│   ├── classification.py
│   ├── routing.py
│   ├── fraud_feature_engineering.py
│   ├── fraud_detection.py
│   └── fraud_analytics.py
├── part_d/
│   ├── policy_retrieval.py
│   ├── refund_agent.py
│   ├── complaint_summarization.py
│   ├── root_cause_analysis.py
│   ├── fraud_investigation_assistant.py
│   ├── nlp_llm_report.py
│   └── policy_metrics_fix.py
├── part_e/
│   ├── retention_policy.py
│   ├── fraud_prevention_policy.py
│   ├── delivery_improvement_policy.py
│   ├── customer_support_policy.py
│   └── business_recommendations.py
├── dashboard/
│   ├── __init__.py
│   └── build_dashboard.py
├── tests/
│   ├── test_part_a.py
│   ├── test_part_b.py
│   ├── test_part_c.py
│   ├── test_part_d.py
│   └── test_part_e.py
└── models/
    ├── .gitkeep
    ├── part_b/
    └── part_c/
```

Generated CSVs, JSON reports, database files, and trained model
artifacts are created by running the scripts. Do not create empty
`.joblib` files manually.

## Generate synthetic data

Run from the repository root:

``` bash
python generate_data.py
```

The supplied generator creates raw CSV files and policy text files under
`data/`. Do not modify the supplied generator.

## Execution guide

Run commands from the repository root, in the order shown. Each stage
may depend on files created by earlier stages.

### Part A --- Data ingestion and quality

``` bash
python part_a/ingestion.py
python part_a/cleaning.py
python part_a/validation.py
python part_a/database.py
python part_a/data_quality_report.py
```

Ingests, cleans, validates, and stores the generated records, then
produces a data-quality report.

### Part B --- Customer intelligence and churn

``` bash
python part_b/feature_engineering.py
python part_b/segmentation.py
python part_b/churn_target.py
python part_b/churn_model_comparison.py
python part_b/customer_intelligence_report.py
```

Creates customer features and segments, prepares churn snapshots,
compares models, and generates the customer intelligence report.

**Churn limitation:** The monthly churn target is an operational proxy.
A customer is labelled as churned when they place no order in the 30
days after a snapshot date. It is not a ground-truth churn label. The
same customer may appear in multiple snapshots; use a chronological
holdout when interpreting evaluation results.

### Part C --- Support tickets and fraud analytics

``` bash
python part_c/ticket_cleaning.py
python part_c/pii_masking.py
python part_c/classification.py
python part_c/routing.py
python part_c/fraud_feature_engineering.py
python part_c/fraud_detection.py
```

Prepares support tickets, masks PII, classifies and routes tickets,
engineers fraud-related features, detects anomalies, and generates fraud
analytics.

**Fraud limitation:** Isolation Forest flags unusual patterns for
further investigation. An anomaly is not proof of fraud.

### Part D --- Policy retrieval and investigation

``` bash
python part_d/policy_retrieval.py
python part_d/refund_agent.py
python part_d/complaint_summarization.py
python part_d/root_cause_analysis.py
python part_d/fraud_investigation_assistant.py
python part_d/policy_metrics_fix.py
python part_d/nlp_llm_report.py
```

Evaluates policy retrieval, processes refund scenarios, summarizes
complaints, analyses potential root causes, and prepares fraud
investigation notes. Run the metrics-fix script before the consolidated
NLP/LLM report.

The investigation assistant supports human review; it does not make
final fraud determinations.

Run fraud analytics after part D run

### Part E --- Business policies and recommendations

``` bash
python part_e/retention_policy.py
python part_e/fraud_prevention_policy.py
python part_e/delivery_improvement_policy.py
python part_e/customer_support_policy.py
python part_e/business_recommendations.py
```

Generates recommendations for customer retention, fraud prevention,
delivery improvement, customer support, and overall business actions.

### Integrated dashboard

After all required reports have been generated:

``` bash
python dashboard/build_dashboard.py
```

Open `data/reports/instamart_dashboard.html` in a web browser.

## Models directory

The `models/` folder stores trained model artifacts:

-   `part_b/churn_model_comparison.py` saves churn model artifacts under
    `models/part_b/`.
-   `part_c/fraud_detection.py` saves the anomaly detection model under
    `models/part_c/`.

The exact artifact filenames are determined by the save paths in those
scripts. The folders can remain empty before execution. Keep `.gitkeep`
if you want Git to retain the otherwise-empty models directory.

## Six final deliverables

  -----------------------------------------------------------------------------------------------------------
  \#                      Deliverable             Expected output
  ----------------------- ----------------------- -----------------------------------------------------------
  1                       Data Quality Report     `data/processed/data_quality_report.json`

  2                       Customer Intelligence   `data/processed/part_b/customer_intelligence_report.json`
                          Report                  

  3                       Fraud Analytics Report  `data/processed/part_c/fraud_analytics_report.json`

  4                       NLP and LLM Report      `data/processed/part_d/nlp_llm_report.json`

  5                       Business Policy Report  `data/processed/part_e/business_policy_report.json`

  6                       Integrated Dashboard    `data/reports/instamart_dashboard.html`
  -----------------------------------------------------------------------------------------------------------

Supporting CSVs, metrics JSON files, SQLite databases, and model files
may also be generated.

## Run tests

Run the full test suite:

``` bash
python -m unittest discover -s tests -v
```

Run only Part E tests:

``` bash
python -m unittest discover -s tests -p "test_part_e.py" -v
```

If a test requires generated data or reports, run the relevant project
stages first.

## Reproducibility and limitations

-   The dataset is synthetic and may not represent actual Instamart
    customers or operations.
-   Churn is a derived 30-day inactivity proxy, not a supplied business
    label.
-   Fraud detection is unsupervised anomaly detection; flagged records
    require human investigation.
-   Policy retrieval and refund decisions are limited to the supplied
    policy documents and synthetic scenarios.
-   PII masking is a demonstration, not a production-grade privacy
    guarantee.
-   Performance on synthetic data does not establish expected real-world
    model performance.
-   Recommendations should be validated against real operational
    evidence before implementation.

## GitHub submission notes

-   Include source code, tests, `README.md`, `requirements.txt`, and the
    supplied generator and mock LLM files.
-   Keep the supplied `generate_data.py` and `mock_llm.py` unchanged.
-   Do not commit secrets, API keys, or real customer information.
-   Generated data, SQLite databases, reports, and model binaries can
    generally be excluded if the assignment expects reproducible source
    code rather than generated outputs. Follow the assignment's explicit
    submission rules if they differ.
-   Keep the complete project in one repository on the required `main`
    branch.

## Quick-start checklist

1.  Open or clone the repository.
2.  Install dependencies from `requirements.txt`.
3.  Run `python generate_data.py`.
4.  Execute Parts A through E in order.
5.  Build the dashboard.
6.  Run the unit tests.
7.  Verify the six final deliverables.
8.  Commit and push the project source and README to GitHub.
