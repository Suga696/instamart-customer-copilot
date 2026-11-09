import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/"data"/"processed"
def main():
    summary=json.loads((PROC/"part_d"/"complaint_summary_report.json").read_text(encoding="utf-8"))
    policy=json.loads((PROC/"part_d"/"policy_evaluation.json").read_text(encoding="utf-8"))
    routing=json.loads((PROC/"part_c"/"routing_report.json").read_text(encoding="utf-8"))
    report={
        "title":"NLP and LLM Report",
        "ticket_analysis":summary,
        "ticket_routing":routing,
        "policy_retrieval_evaluation":policy,
        "rag_design":{
            "documents":["refund_policy.txt","delivery_policy.txt","payment_policy.txt","privacy_policy.txt","escalation_policy.txt"],
            "retrieval_method":"Token overlap with inverse-document-frequency weighting; top relevant policy documents passed as context to Mock LLM POLICY ASSISTANT mode.",
            "abstention":"The assistant abstains when retrieval evidence is absent or below its configured threshold."
        },
        "limitations":[
            "The supplied mock_llm supports classification, policy Q&A and agent modes, but no generative complaint summarisation mode.",
            "Complaint summary and root-cause analysis are deterministic data aggregations, not LLM-generated causal conclusions.",
            "Predicted ticket categories are not independently verified ground truth."
        ]
    }
    (PROC/"part_d"/"nlp_llm_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("NLP and LLM Report generated.")
if __name__=="__main__": main()
