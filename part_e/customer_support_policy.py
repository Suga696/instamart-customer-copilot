"""Generate support operations recommendations using actual routing outputs."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"processed"/"part_e"
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(ROOT/"data"/"processed"/"part_c"/"routed_tickets.csv")
    queues=r.groupby(["assigned_queue","priority"]).size().rename("ticket_count").reset_index()
    queues.to_csv(OUT/"support_queue_summary.csv",index=False)
    policy={
        "routing_summary":queues.to_dict(orient="records"),
        "rules":[
            "Route billing, delivery and technical complaints to their corresponding support queues.",
            "Route fraud or unauthorised-transaction complaints to Security Team with high priority and a two-hour response target, as stated in the supplied escalation policy.",
            "Send unclassified, uncertain or unmatched tickets to Manual Review.",
            "Preserve the one-hour premium and 24-hour standard response targets only when customer segment membership is verified.",
            "Use approved response templates and escalate unresolved cases to a human agent."
        ],
        "limitation":"The supplied K-Means cluster IDs have no authoritative Premium mapping; no cluster is assumed to be Premium."
    }
    (OUT/"customer_support_policy.json").write_text(json.dumps(policy,indent=2),encoding="utf-8")
    print("Customer support policy generated.")
if __name__=="__main__": main()
