"""Recalculate policy abstention metrics from the persisted CSV using boolean-safe parsing."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"processed"/"part_d"
def main():
    rows=list(csv.DictReader((OUT/"policy_answers.csv").open(encoding="utf-8",newline="")))
    answerable=[r for r in rows if r["expected_doc"]!="NONE"]
    unanswerable=[r for r in rows if r["expected_doc"]=="NONE"]
    doc=sum(r["expected_doc"] in (r["source_documents"] or "").split(";") for r in answerable)
    fact=sum(bool(r["expected_fact"]) and r["expected_fact"].lower() in r["answer"].lower() for r in answerable)
    abstain=sum(str(r["abstained"]).strip().lower()=="true" for r in unanswerable)
    result={
        "total_questions":len(rows),"answerable_questions":len(answerable),
        "unanswerable_questions":len(unanswerable),
        "expected_document_retrieval_count":doc,"expected_fact_in_answer_count":fact,
        "correct_abstentions":abstain,
        "document_retrieval_accuracy":round(doc/len(answerable),4) if answerable else 0,
        "fact_answer_match_rate":round(fact/len(answerable),4) if answerable else 0,
        "unanswerable_abstention_rate":round(abstain/len(unanswerable),4) if unanswerable else 0,
        "note":"Recalculated from policy_answers.csv with boolean-safe abstention parsing."
    }
    (OUT/"policy_evaluation_corrected.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
