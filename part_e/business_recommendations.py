import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data"/"processed"/"part_e"
def main():
    names=["retention_policy.json","fraud_prevention_policy.json",
           "delivery_improvement_policy.json","customer_support_policy.json"]
    content={}
    for name in names:
        p=BASE/name
        if not p.exists(): raise FileNotFoundError(f"Run the corresponding Part E script first: {p}")
        content[name.replace(".json","")]=json.loads(p.read_text(encoding="utf-8"))
    report={
        "title":"Instamart Evidence-Based Business Policy Report",
        "recommendations":content,
        "governance":[
            "Treat model outputs as decision support, not automatic customer decisions.",
            "Use human review for suspected fraud and exceptional refund/account actions.",
            "Measure interventions using holdout or before/after designs where feasible.",
            "Separate observed descriptive patterns from causal claims.",
            "Document data gaps and review model performance as new data becomes available."
        ]
    }
    (BASE/"business_policy_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Consolidated business policy report generated.")
if __name__=="__main__": main()
