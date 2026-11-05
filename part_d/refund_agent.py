"""
Part D - Tool-using refund support agent.

Run from the repository root:
    python part_d/refund_agent.py

Reads:
    data/database/instamart.db
    data/policies/*.txt
    data/raw/agent_scenarios.csv

Writes:
    data/processed/part_d/refund_agent_results.csv
    data/processed/part_d/refund_agent_metrics.json

The supplied mock_llm.py drives the agent's tool-call sequence. All actions
are simulated locally; this script does not contact a payment provider or
issue a real refund.
"""

import csv
import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "instamart.db"
POLICY_DIR = PROJECT_ROOT / "data" / "policies"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "part_d"
MAX_AGENT_STEPS = 6


def read_policy_text():
    """Load policy documents for the agent's policy-search tool."""
    policy_text = {}
    for path in sorted(POLICY_DIR.glob("*.txt")):
        policy_text[path.name] = path.read_text(encoding="utf-8").strip()
    if not policy_text:
        raise FileNotFoundError(
            f"No policy documents found in {POLICY_DIR}. "
            "Run generate_data.py first."
        )
    return policy_text


def get_order(connection, order_id):
    """Fetch the exact order fields used by the refund policy and mock agent."""
    row = connection.execute(
        """
        SELECT order_id, customer_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,),
    ).fetchone()

    if row is None:
        return {"error": "order_not_found", "order_id": order_id}

    return {
        "order_id": row[0],
        "customer_id": row[1],
        "status": row[2],
        "order_amount": row[3],
        "delivery_minutes": row[4],
    }


def execute_tool(tool_name, arguments, connection, policy_text, action_log):
    """Execute one allow-listed local tool requested by mock_llm."""
    if tool_name == "get_order":
        return get_order(connection, arguments.get("order_id", ""))

    if tool_name == "search_policy":
        query = str(arguments.get("query", "")).lower()
        # Return the refund rules because this tool is specifically used
        # in the supplied refund-agent workflow.
        refund_policy = policy_text.get("refund_policy.txt", "")
        if not refund_policy:
            return {"error": "refund_policy_missing"}
        action_log.append("search_policy")
        return {
            "document": "refund_policy.txt",
            "query": query,
            "policy_excerpt": refund_policy,
        }

    if tool_name == "calculate_refund":
        amount = float(arguments.get("order_amount", 0) or 0)
        delivery_minutes = arguments.get("delivery_minutes")
        if delivery_minutes is None:
            return {"error": "delivery_duration_unavailable", "refund_inr": 0}
        refund_amount = amount if int(delivery_minutes) > 45 else 0.0
        return {
            "refund_inr": round(refund_amount, 2),
            "calculation_rule": "full order amount when delivery exceeds 45 minutes",
        }

    if tool_name == "issue_refund":
        # Simulation only. No external payment transaction is performed.
        refund_amount = round(float(arguments.get("refund_inr", 0)), 2)
        action_log.append("simulated_refund")
        return {
            "status": "simulated",
            "order_id": arguments.get("order_id"),
            "refund_inr": refund_amount,
        }

    if tool_name == "escalate_to_human":
        action_log.append("human_escalation")
        return {
            "status": "escalated",
            "order_id": arguments.get("order_id"),
            "refund_inr": round(float(arguments.get("refund_inr", 0)), 2),
            "assigned_to": "human support lead",
        }

    return {"error": f"unsupported_tool: {tool_name}"}


def run_agent(message, connection, policy_text, mock_llm, action_log):
    """Run a bounded ReAct-style tool loop using the supplied offline mock."""
    system_prompt = (
        "AGENT. You are an Instamart refund support agent. "
        "Use the available tools to verify an order and apply the refund policy. "
        "Never claim a real payment has been made."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    for _step in range(MAX_AGENT_STEPS):
        response = mock_llm.chat(messages, tools=[])
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if not tool_calls:
            return assistant_message.content or "FALLBACK: no response from agent."

        # Preserve the assistant tool-call message in the conversation.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in tool_calls
            ],
        })

        for call in tool_calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            result = execute_tool(
                call.name, arguments, connection, policy_text, action_log
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": json.dumps(result),
            })

    return "FALLBACK: maximum agent tool steps reached; hand over to a human agent."


def outcome_from_response(response_text):
    """Extract the stable outcome label returned by the supplied mock agent."""
    text = (response_text or "").strip()
    for label in ("REFUND", "ESCALATE", "DENIED", "NOT_FOUND", "NEEDS_INFO", "FALLBACK"):
        if text.startswith(label + ":") or text.startswith(label + " "):
            return label
    return "FALLBACK"


def main():
    try:
        import mock_llm
    except ImportError as exc:
        raise ImportError(
            "Could not import mock_llm.py. Keep the supplied file unchanged "
            "in the repository root."
        ) from exc

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Missing database: {DATABASE_PATH}. "
            "Run Part A through 'python part_a/database.py' first."
        )

    scenario_path = RAW_DIR / "agent_scenarios.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Missing {scenario_path}. Run generate_data.py first."
        )

    policy_text = read_policy_text()
    with scenario_path.open("r", newline="", encoding="utf-8") as file:
        scenarios = list(csv.DictReader(file))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    results = []

    try:
        # Reset deterministic mock counters once for this independent batch.
        if hasattr(mock_llm, "reset"):
            mock_llm.reset()

        for index, scenario in enumerate(scenarios, start=1):
            action_log = []
            response_text = run_agent(
                scenario["message"],
                connection,
                policy_text,
                mock_llm,
                action_log,
            )
            actual_outcome = outcome_from_response(response_text)
            expected_outcome = scenario["expected_outcome"].strip()
            try:
                expected_refund = float(scenario["expected_refund_inr"])
            except (TypeError, ValueError):
                expected_refund = 0.0

            # The mock's response includes the amount for completed refund
            # outcomes. Parse it only for reporting; never execute a payment.
            actual_refund = 0.0
            marker = "INR "
            if actual_outcome == "REFUND" and marker in response_text:
                amount_text = response_text.split(marker, 1)[1].split()[0]
                try:
                    actual_refund = float(amount_text.replace(",", "").rstrip("."))
                except ValueError:
                    actual_refund = 0.0

            results.append({
                "scenario_id": scenario["scenario_id"],
                "customer_id": scenario["customer_id"],
                "message": scenario["message"],
                "expected_outcome": expected_outcome,
                "actual_outcome": actual_outcome,
                "expected_refund_inr": expected_refund,
                "actual_refund_inr": actual_refund,
                "outcome_match": actual_outcome == expected_outcome,
                "response": response_text,
                "tools_used": ";".join(action_log),
            })
            print(
                f"[{index}/{len(scenarios)}] {scenario['scenario_id']}: "
                f"expected={expected_outcome}, actual={actual_outcome}"
            )
            print(f"  {response_text}")

    finally:
        connection.close()

    output_csv = OUTPUT_DIR / "refund_agent_results.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    matched = sum(bool(row["outcome_match"]) for row in results)
    metrics = {
        "total_scenarios": total,
        "outcome_matches": matched,
        "outcome_accuracy": round(matched / total, 4) if total else 0.0,
        "expected_outcomes": {
            label: sum(row["expected_outcome"] == label for row in results)
            for label in sorted({row["expected_outcome"] for row in results})
        },
        "actual_outcomes": {
            label: sum(row["actual_outcome"] == label for row in results)
            for label in sorted({row["actual_outcome"] for row in results})
        },
        "note": (
            "Refund and escalation actions are local simulations only. "
            "No real payment is issued."
        ),
    }

    metrics_path = OUTPUT_DIR / "refund_agent_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nRefund-agent evaluation:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results: {output_csv.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {metrics_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
