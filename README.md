Instamart Customer Intelligence & Support Copilot

An AI-powered customer intelligence and support solution built using Python, Machine Learning, NLP, SQLite and a simulated LLM workflow.

1. Project Overview

The Instamart Customer Intelligence & Support Copilot is a data-driven solution designed to improve customer understanding, identify potential customer churn, automate support ticket classification, and assist customer support teams with policy-based responses and refund processing.

The project uses synthetic customer, product, order and support ticket datasets to build an integrated customer intelligence and support workflow.

It is divided into four connected parts:

Part A – Data Engineering: Data ingestion, cleaning, validation and database creation.

Part B – Customer Intelligence: Customer feature engineering, segmentation and churn prediction.

Part C – Support Ticket Intelligence: Ticket preprocessing, PII masking, classification and routing.

Part D – AI Support Copilot: Policy retrieval, policy question answering and refund agent workflows.

2. Business Problem

An online grocery delivery platform handles a large volume of customer orders and support requests. Without an integrated customer intelligence system, it can be difficult to:

Maintain consistent and reliable customer and order data.

Understand customer purchasing behaviour.

Identify customers who may be at risk of churn.

Process large volumes of customer support tickets efficiently.

Protect personally identifiable information (PII).

Route customer complaints to the appropriate support teams.

Provide consistent answers based on company policies.

Handle refund requests according to predefined business rules.

This project addresses these challenges through a connected data engineering, machine learning and AI-assisted support workflow.

3. Project Objectives

Build a trusted analytical database from raw synthetic datasets.

Clean and validate customer, product and order information.

Develop customer-level analytical features.

Segment customers based on purchasing behaviour.

Build a machine learning model to predict customer churn.

Preprocess support tickets and mask sensitive information.

Classify tickets into appropriate support categories.

Route tickets based on category and customer segment information.

Retrieve relevant policy documents to answer customer questions.

Develop a simulated refund agent with controlled business-rule execution.

4. Technology Stack

Technology

Purpose

Python 3.11+

Core programming language

Pandas

Data manipulation and preprocessing

NumPy

Numerical operations

Scikit-learn

Customer segmentation and churn prediction

Joblib

Saving and loading machine learning models

SQLite

Analytical database

Python unittest

Unit testing

Mock LLM

Simulated language model interactions

Git and GitHub

Version control and project repository

Note: The project uses the supplied generate_data.py and mock_llm.py files. The mock LLM supports offline execution without requiring an external API key.

5. Project Architecture

The project follows a four-part workflow.

Part A – Data Engineering

Raw customer, product and order datasets are processed to create a reliable analytical database.

Activities:

Data ingestion and profiling.

Missing-value and duplicate handling.

Data type and format standardisation.

Invalid value identification and cleaning.

Primary key and foreign key validation.

SQLite database creation.

Database tables, indexes and analytical view creation.

Expected output:

Cleaned customer, product and order datasets.

Data quality and validation reports.

SQLite analytical database.

Part B – Customer Intelligence

Customer-level features are used to understand purchasing patterns and predict potential churn.

Activities:

Customer-level feature engineering.

Aggregation of order and purchase information.

Customer segmentation using K-Means clustering.

Cluster profiling.

Churn prediction using a supervised machine learning model.

Model evaluation using appropriate classification metrics.

Expected output:

Customer feature dataset.

Customer segmentation results.

Segment profile report.

Saved segmentation model.

Churn predictions and evaluation metrics.

Saved churn prediction model.

Important: The churn target must follow the exact definition provided in the assignment. Cluster numbers are algorithm-generated identifiers and should not automatically be interpreted as customer labels such as Premium or Regular.

Part C – Support Ticket Intelligence

Customer support tickets are processed to identify the issue category and route tickets to the appropriate support team.

Activities:

Ticket text cleaning and preprocessing.

Personally Identifiable Information (PII) masking.

Ticket classification using the supplied Mock LLM.

Handling transient failures, malformed responses and unsupported categories.

Customer segment integration.

Ticket priority and support-team routing.

Manual review handling for uncertain or unclassified tickets.

Ticket categories:

Billing

Delivery

Technical

Fraud

Expected output:

Cleaned support tickets.

PII-masked ticket dataset.

Classified ticket dataset.

Routed ticket dataset.

Classification and routing reports.

Part D – AI Support Copilot

This part provides policy-based question answering and a simulated refund-processing workflow.

Activities:

Loading company policy documents.

Retrieving relevant policy information.

Answering customer policy questions using the supplied Mock LLM.

Evaluating policy retrieval and responses.

Retrieving customer order details from the Part A database.

Calculating eligible refunds based on policy rules.

Simulating refund execution.

Escalating requests that require human intervention.

Handling missing orders and unsupported requests.

Expected output:

Policy question-answering results.

Policy evaluation report.

Refund agent execution results.

Refund scenario evaluation report.

6. Repository Structure

instamart-customer-copilot/
│
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
│
├── part_a/
│   ├── ingestion.py
│   ├── cleaning.py
│   ├── validation.py
│   └── database.py
│
├── part_b/
│   ├── feature_engineering.py
│   ├── segmentation.py
│   └── churn_prediction.py
│
├── part_c/
│   ├── ticket_cleaning.py
│   ├── pii_masking.py
│   ├── classification.py
│   └── routing.py
│
├── part_d/
│   ├── policy_retrieval.py
│   └── refund_agent.py
│
├── models/
│   ├── part_b/
│   └── part_d/
│
└── tests/
    ├── __init__.py
    ├── test_part_a.py
    ├── test_part_b.py
    ├── test_part_c.py
    └── test_part_d.py

Generated datasets, reports, database files and models are stored in their respective directories during execution.

7. Installation and Setup

Prerequisites

Python 3.11 or above.

SQLite 3.25 or above.

Git.

GitHub account.

Step 1 – Clone the repository

git clone https://github.com/<your-username>/instamart-customer-copilot.git
cd instamart-customer-copilot

Step 2 – Create a virtual environment

python -m venv venv

Activate the environment.

Windows:

venv\Scripts\activate

Linux / macOS:

source venv/bin/activate

Step 3 – Install dependencies

pip install -r requirements.txt

The project dependencies are:

pandas
numpy
scikit-learn
joblib

8. Execution Instructions

Execute the scripts from the repository root directory in the following order.

Step 1 – Generate synthetic datasets

python generate_data.py

This generates synthetic customer, product, order and support ticket datasets along with policy and evaluation files.

Step 2 – Execute Part A

python part_a/ingestion.py
python part_a/cleaning.py
python part_a/validation.py
python part_a/database.py

This creates cleaned datasets, validation reports and the SQLite analytical database.

Step 3 – Execute Part B

python part_b/feature_engineering.py
python part_b/segmentation.py
python part_b/churn_prediction.py

This creates customer-level features, customer segments and churn prediction results.

Step 4 – Execute Part C

python part_c/ticket_cleaning.py
python part_c/pii_masking.py
python part_c/classification.py
python part_c/routing.py

This cleans and masks support ticket information, classifies customer issues and routes tickets to appropriate support teams.

Step 5 – Execute Part D

python part_d/policy_retrieval.py
python part_d/refund_agent.py

This executes policy-based question answering and refund agent scenarios.

Step 6 – Execute unit tests

python -m unittest discover -s tests -v

This runs the available unit tests for all four project parts.

9. Expected Outputs

Project Part

Expected outputs

Part A

Cleaned CSV files, ingestion report, validation report and SQLite database

Part B

Customer features, customer segments, segment profiles, churn predictions and model metrics

Part C

Cleaned tickets, masked tickets, classified tickets, routed tickets and reports

Part D

Policy answers, policy evaluation results and refund agent scenario results

The generated files are stored under:

data/processed/
data/database/
models/

Exact output filenames are defined by the corresponding Python modules.

10. Data Quality and Model Evaluation

The project includes validation and evaluation activities across the four parts.

Part A – Data Quality

Missing-value checks.

Duplicate record checks.

Data type validation.

Primary key validation.

Foreign key validation.

Business-rule validation.

Part B – Machine Learning

Customer segmentation evaluation using silhouette score.

Churn prediction evaluation using accuracy, precision, recall, F1-score and ROC-AUC, where applicable.

Part C – Ticket Classification

Valid category verification.

Invalid response handling.

Retry and fallback handling.

PII masking checks.

Ticket routing validation.

Part D – Policy and Refund Evaluation

Policy document retrieval evaluation.

Expected fact verification.

Abstention handling for unsupported questions.

Refund scenario outcome verification.

Human escalation and missing-order handling.

11. Reproducibility

The project is designed to support repeatable execution.

Synthetic data is generated using the supplied generator.

Fixed random seeds are used in the applicable machine learning workflows.

The supplied Mock LLM is used for offline language-model simulation.

No external LLM API key is required.

Generated reports and model artifacts are saved locally.

Execution commands are documented in this README.

12. Key Learning Outcomes

Through this project, the following concepts are implemented:

End-to-end data engineering.

Data cleaning and validation.

Relational database design using SQLite.

Customer analytics and feature engineering.

Unsupervised learning using K-Means.

Supervised learning for churn prediction.

Natural language processing workflows.

PII masking and data privacy.

AI-assisted support ticket classification.

Policy document retrieval.

Tool-based simulated refund workflows.

Unit testing and reproducible project execution.

GitHub-based project management.

13. Future Enhancements

Potential future enhancements include:

Real-time customer activity ingestion.

Advanced customer behaviour analytics.

Improved churn prediction and model monitoring.

Additional support ticket categories.

More advanced policy retrieval methods.

Integration with a real customer support platform.

Dashboard-based monitoring of customer intelligence metrics.

14. Disclaimer

This project uses synthetic datasets and a simulated LLM workflow for educational and demonstration purposes. Refund processing is simulated and does not initiate actual financial transactions.

Project: Instamart Customer Intelligence & Support Copilot
Repository: instamart-customer-copilot
Branch: main
Language: Python
Purpose: End-to-end Data Engineering, Machine Learning and AI-assisted Customer Support
