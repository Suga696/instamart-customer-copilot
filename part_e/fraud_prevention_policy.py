"""Translate anomaly-screening outputs into human-review policy guidance."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"processed"/"part_e"
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(ROOT/"data"/"processed"/"part_c"/"fraud_anomaly_results.csv")
    counts=f.review_priority.value_counts().to_dict()
    policy={
        "LOW":"Continue normal processing and routine monitoring.",
        "MEDIUM":"Apply proportionate additional verification where appropriate; do not delay ordinary service without evidence.",
        "HIGH":"Prioritise manual review before exceptional refund or account action; document evidence and provide an appeal route.",
        "safeguards":[
            "An anomaly score is not proof of fraud.",
            "Never reject a refund or penalise a customer solely because of a model score.",
            "A trained human reviewer must examine transaction and complaint evidence.",
            "Record review outcomes and monitor false positives and disparate impact where data permits."
        ],
        "screening_counts":{str(k):int(v) for k,v in counts.items()},
        "label_limitation":"No verified fraud labels are supplied; supervised fraud performance cannot be estimated."
    }
    (OUT/"fraud_prevention_policy.json").write_text(json.dumps(policy,indent=2),encoding="utf-8")
    pd.DataFrame([{"priority":k,"orders":int(v)} for k,v in counts.items()]).to_csv(OUT/"fraud_review_policy_counts.csv",index=False)
    print("Fraud prevention policy generated.")
if __name__=="__main__": main()
