import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "part_c"
MODEL_DIR = ROOT / "models" / "part_c"

def main():
    path = OUT / "fraud_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run part_c/fraud_feature_engineering.py first.")
    df = pd.read_csv(path)
    excluded = {"order_id", "customer_id", "product_id", "order_ts", "status"}
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    if len(df) < 10:
        raise ValueError("At least 10 order records are required for anomaly screening.")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("isolation_forest", IsolationForest(
            n_estimators=250, contamination=0.03, random_state=42
        )),
    ])
    model.fit(df[features])
    prediction = model.predict(df[features])
    anomaly_score = -model.decision_function(df[features])
    output = df[["order_id", "customer_id", "order_ts", "status"]].copy()
    output["anomaly_flag"] = (prediction == -1)
    output["anomaly_score"] = anomaly_score
    # Risk bands are review-priority labels, not determinations of fraud.
    q90, q98 = np.quantile(anomaly_score, [0.90, 0.98])
    output["review_priority"] = np.where(
        output["anomaly_flag"],
        np.where(anomaly_score >= q98, "HIGH", "MEDIUM"),
        "LOW",
    )
    output["recommended_action"] = np.where(
        output.anomaly_flag, "Human review; verify order and customer evidence",
        "Normal processing; retain routine monitoring"
    )
    output.to_csv(OUT / "fraud_anomaly_results.csv", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features}, MODEL_DIR / "isolation_forest.joblib")
    report = {
        "method": "Isolation Forest",
        "rows_scored": int(len(output)),
        "anomalies_flagged": int(output.anomaly_flag.sum()),
        "anomaly_rate": float(output.anomaly_flag.mean()),
        "high_priority": int((output.review_priority == "HIGH").sum()),
        "medium_priority": int((output.review_priority == "MEDIUM").sum()),
        "features": features,
        "contamination_setting": 0.03,
        "risk_thresholds": {"medium_percentile": 90, "high_percentile": 98,
                            "medium_cutoff": float(q90), "high_cutoff": float(q98)},
        "important_caveat": "An anomaly is not proof of fraud. All flagged records require human investigation; no customer is automatically penalised.",
        "label_limitation": "No verified fraud labels are present, so supervised fraud metrics cannot be calculated.",
    }
    (OUT / "fraud_detection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Isolation Forest flagged {report['anomalies_flagged']} of {len(output)} orders for review.")

if __name__ == "__main__":
    main()
