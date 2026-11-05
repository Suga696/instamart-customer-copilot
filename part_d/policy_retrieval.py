"""
Part D - Policy document retrieval and grounded Q&A.

Run from the repository root:
    python part_d/policy_retrieval.py

Reads:
    data/policies/*.txt
    data/raw/policy_eval.csv

Writes:
    data/processed/part_d/policy_answers.csv
    data/processed/part_d/policy_evaluation.json

Uses the supplied, unchanged mock_llm.py in POLICY ASSISTANT mode.
Only retrieved policy text is sent to the mock language model. Questions
with no meaningful policy match are answered with an explicit abstention.
"""

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_d"
POLICY_FILES = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
    "escalation_policy.txt",
]
ABSTENTION = "I could not find that information in the policy documents."
STOP_WORDS = {
    "the", "a", "an", "is", "of", "to", "for", "in", "on", "how", "what",
    "when", "who", "are", "do", "does", "i", "my", "be", "can", "after",
    "from", "it", "or", "and", "at", "by", "which", "why", "if", "than",
    "that", "with", "must", "should", "would", "will", "have", "has",
}


def tokenize(text):
    """Return normalized terms, excluding common stop words."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def load_policy_documents():
    """Load the supplied policy text files without changing their contents."""
    documents = {}
    for filename in POLICY_FILES:
        path = POLICY_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing policy document: {path}. "
                "Run 'python generate_data.py' from the repository root."
            )
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Policy document is empty: {path}")
        documents[filename] = content
    return documents


def retrieve_documents(question, documents, top_k=3):
    """
    Rank policy documents using simple term overlap with inverse-document
    frequency weighting. This is transparent retrieval, not an LLM guess.
    """
    query_terms = set(tokenize(question))
    if not query_terms:
        return [], 0.0

    document_terms = {
        name: set(tokenize(content)) for name, content in documents.items()
    }
    document_frequency = {
        term: sum(term in terms for terms in document_terms.values())
        for term in query_terms
    }

    ranked = []
    for name, terms in document_terms.items():
        matched = query_terms & terms
        score = sum(
            1.0 + (1.0 / document_frequency[term])
            for term in matched
        )
        if matched:
            ranked.append((name, score, sorted(matched)))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_k], (ranked[0][1] if ranked else 0.0)


def ask_policy_question(question, documents, mock_llm):
    """Retrieve context, ask the offline mock LLM, and return its answer."""
    ranked, best_score = retrieve_documents(question, documents)

    # An explicit threshold prevents unrelated questions being answered
    # from a weak keyword match. The supplied evaluation's two NONE
    # questions have no meaningful overlap with the policy corpus.
    if not ranked or best_score < 2.0:
        return {
            "answer": ABSTENTION,
            "source_documents": [],
            "retrieval_score": best_score,
            "abstained": True,
        }

    context_parts = []
    source_documents = []
    for filename, score, _matched_terms in ranked:
        context_parts.append(f"[{filename}] {documents[filename]}")
        source_documents.append(filename)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer only from the supplied policy "
                "context. Do not invent rules. If the context does not "
                "contain the answer, say that the information was not found."
            ),
        },
        {
            "role": "user",
            "content": (
                "CONTEXT:\n" + "\n".join(context_parts)
                + "\nQUESTION:\n" + question
            ),
        },
    ]
    response = mock_llm.chat(messages)
    answer = response.choices[0].message.content or ABSTENTION

    return {
        "answer": answer.strip(),
        "source_documents": source_documents,
        "retrieval_score": best_score,
        "abstained": answer.strip() == ABSTENTION,
    }


def evaluate_answers(answer_rows):
    """Compare generated answers with the supplied expected document/fact."""
    total = len(answer_rows)
    doc_correct = 0
    fact_found = 0
    abstention_correct = 0

    for row in answer_rows:
        expected_doc = row["expected_doc"]
        expected_fact = (row["expected_fact"] or "").strip().lower()
        sources = row["source_documents"].split(";") if row["source_documents"] else []
        answer = row["answer"].lower()

        if expected_doc == "NONE":
            if row["abstained"] == "True":
                abstention_correct += 1
        else:
            if expected_doc in sources:
                doc_correct += 1
            if expected_fact and expected_fact in answer:
                fact_found += 1

    answerable = sum(row["expected_doc"] != "NONE" for row in answer_rows)
    unanswerable = total - answerable
    return {
        "total_questions": total,
        "answerable_questions": answerable,
        "unanswerable_questions": unanswerable,
        "expected_document_retrieval_count": doc_correct,
        "expected_fact_in_answer_count": fact_found,
        "correct_abstentions": abstention_correct,
        "document_retrieval_accuracy": round(doc_correct / answerable, 4) if answerable else 0.0,
        "fact_answer_match_rate": round(fact_found / answerable, 4) if answerable else 0.0,
        "unanswerable_abstention_rate": round(abstention_correct / unanswerable, 4) if unanswerable else 0.0,
        "note": (
            "Metrics are descriptive checks against the supplied evaluation "
            "file; no target score is assumed."
        ),
    }


def main():
    try:
        import mock_llm
    except ImportError as exc:
        raise ImportError(
            "Could not import mock_llm.py. Keep the supplied file unchanged "
            "in the repository root."
        ) from exc

    documents = load_policy_documents()
    evaluation_path = RAW_DIR / "policy_eval.csv"
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Missing {evaluation_path}. Run generate_data.py first."
        )

    with evaluation_path.open("r", newline="", encoding="utf-8") as file:
        questions = list(csv.DictReader(file))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for index, item in enumerate(questions, start=1):
        result = ask_policy_question(item["question"], documents, mock_llm)
        results.append({
            "question": item["question"],
            "expected_doc": item["expected_doc"],
            "expected_fact": item["expected_fact"],
            "answer": result["answer"],
            "source_documents": ";".join(result["source_documents"]),
            "retrieval_score": round(result["retrieval_score"], 4),
            "abstained": result["abstained"],
        })
        print(f"[{index}/{len(questions)}] {item['question']}")
        print(f"  Answer: {result['answer']}")

    output_csv = OUTPUT_DIR / "policy_answers.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    metrics = evaluate_answers(results)
    metrics_path = OUTPUT_DIR / "policy_evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nPolicy evaluation summary:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved answers: {output_csv.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
