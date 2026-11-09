import json
from pathlib import Path
import pandas as pd
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "part_d"
PROC = ROOT / "data" / "processed"

def top_terms(series, limit=8):
    stop = {"the","and","for","with","that","this","was","are","have","has","from","not","but","you","your","order","instamart"}
    words = []
    for text in series.fillna("").astype(str):
        words.extend(w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop)
    return [w for w, _ in Counter(words).most_common(limit)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tickets = pd.read_csv(PROC / "part_c" / "classified_tickets.csv")
    customers = pd.read_csv(PROC / "customers_clean.csv")
    merged = tickets.merge(customers[["customer_id","city"]], on="customer_id", how="left")
    summary = merged.groupby("predicted_category").agg(
        ticket_count=("ticket_id","count"),
        unique_customers=("customer_id","nunique"),
        cities=("city", lambda x: int(x.nunique())),
    ).reset_index().sort_values("ticket_count", ascending=False)
    summary["share_pct"] = (summary.ticket_count / max(len(merged),1) * 100).round(2)
    summary.to_csv(OUT / "complaint_category_summary.csv", index=False)
    city_summary = merged.groupby(["city","predicted_category"]).size().rename("ticket_count").reset_index()
    city_summary.to_csv(OUT / "complaints_by_city.csv", index=False)
    terms = top_terms(merged.text)
    narrative = [
        f"The dataset contains {len(merged)} support tickets from {merged.customer_id.nunique()} customers.",
        "Most frequent predicted categories: " + ", ".join(
            f"{r.predicted_category} ({r.ticket_count})" for r in summary.head(3).itertuples()
        ) + ".",
        "Frequent text terms (descriptive, not causal): " + ", ".join(terms) + ".",
        "Category labels are Mock LLM predictions; empty descriptions and fallback cases remain explicitly marked.",
    ]
    report = {
        "ticket_count": int(len(merged)),
        "unique_customers": int(merged.customer_id.nunique()),
        "category_summary": summary.to_dict(orient="records"),
        "frequent_terms": terms,
        "evidence_based_summary": " ".join(narrative),
        "generation_method": "Deterministic aggregation and extractive term counts.",
        "llm_limitation": "The supplied mock_llm.py supports CLASSIFY, POLICY ASSISTANT and AGENT modes, but no complaint-summarisation mode. To preserve the supplied file unchanged, this report does not claim that an LLM generated the summary.",
        "limitations": ["Ticket sample is synthetic and small.", "Text terms and category counts describe associations, not root causes."],
    }
    (OUT / "complaint_summary_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["evidence_based_summary"])

if __name__ == "__main__":
    main()
