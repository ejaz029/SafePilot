
"""Generate runtime-backed SafePilot multi-step trajectories."""

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.benchmark_adapter import adapt_scenario
from environment.business_store import BusinessStore
from environment.payment_ledger import PaymentLedger
from environment.approval_manager import ApprovalManager
from authorization.identity_store import IdentityStore
from authorization.policies import UserIdentity, UserRole
from agent.tools import SafePilotTools


BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "training" / "prepared"
OUTPUT_PATH = OUTPUT_DIR / "runtime_trajectories_train.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "trajectory_manifest.json"

EMPLOYEE_ID = "benchmark_employee"
MANAGER_ID = "benchmark_finance_manager"
ADMIN_ID = "benchmark_finance_admin"

WORKFLOW_CATEGORIES = {
    "normal_workflow",
    "indirect_prompt_injection",
    "conflicting_instructions",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def successful_read(result: Any, expected_key: str) -> bool:
    """Read tools return resource data directly, not success=True."""
    return isinstance(result, dict) and bool(result.get(expected_key))


def successful_action(result: Any) -> bool:
    """Transactional tools explicitly return success=True/False."""
    return isinstance(result, dict) and result.get("success") is True


def create_runtime():
    business_store = BusinessStore()
    payment_ledger = PaymentLedger()
    approval_manager = ApprovalManager()
    identity_store = IdentityStore()

    identity_store.register_user(
        UserIdentity(user_id=EMPLOYEE_ID, role=UserRole.EMPLOYEE)
    )
    identity_store.register_user(
        UserIdentity(
            user_id=MANAGER_ID,
            role=UserRole.FINANCE_MANAGER,
        )
    )
    identity_store.register_user(
        UserIdentity(user_id=ADMIN_ID, role=UserRole.FINANCE_ADMIN)
    )

    tools = SafePilotTools(
        business_store=business_store,
        identity_store=identity_store,
        payment_ledger=payment_ledger,
        approval_manager=approval_manager,
    )

    return business_store, tools


def make_record(
    scenario: dict,
    step: int,
    decision: str,
    tool: str | None,
    arguments: dict,
    observation: Any,
    workflow_state: dict,
    event_type: str = "AGENT_ACTION",
) -> dict:
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario["category"],
        "step": step,
        "event_type": event_type,
        "user_request": scenario["user_request"],
        "invoice_content": scenario.get("invoice_content"),
        "purchase_order": scenario.get("purchase_order"),
        "user_role": scenario["user_role"],
        "decision": decision,
        "tool": tool,
        "arguments": arguments,
        "observation": json_safe(observation),
        "workflow_state": workflow_state.copy(),
    }


def generate_scenario_trajectory(scenario: dict) -> list[dict]:
    if scenario["category"] not in WORKFLOW_CATEGORIES:
        return []

    fixture = adapt_scenario(scenario)

    if fixture.invoice is None or fixture.purchase_order is None:
        return []

    business_store, tools = create_runtime()

    business_store.add_invoice(fixture.invoice)
    business_store.add_purchase_order(fixture.purchase_order)

    invoice_id = fixture.invoice.invoice_id
    po_id = fixture.purchase_order.po_id

    records = []

    state = {
        "invoice_read": False,
        "purchase_order_read": False,
        "payment_status": "NOT_PREPARED",
        "approval_status": "NOT_REQUESTED",
        "execution_status": "NOT_EXECUTED",
    }

    def record_action(step, tool_name, arguments, result):
        records.append(
            make_record(
                scenario=scenario,
                step=step,
                decision="ACT",
                tool=tool_name,
                arguments=arguments,
                observation=result,
                workflow_state=state,
            )
        )

    # 1. Read invoice.
    read_result = tools.read_invoice(invoice_id=invoice_id)

    if not successful_read(read_result, "invoice_id"):
        return records

    state["invoice_read"] = True
    record_action(
        1, "read_invoice", {"invoice_id": invoice_id}, read_result
    )

    # 2. Retrieve purchase order.
    po_result = tools.lookup_purchase_order(po_id=po_id)

    if not successful_read(po_result, "po_id"):
        return records

    state["purchase_order_read"] = True
    record_action(
        2, "lookup_purchase_order", {"po_id": po_id}, po_result
    )

    # 3. Verify and prepare payment.
    prepare_args = {
        "invoice_id": invoice_id,
        "po_id": po_id,
    }

    prepare_result = tools.prepare_payment(
        invoice_id=invoice_id,
        po_id=po_id,
        requester_id=EMPLOYEE_ID,
    )

    if not successful_action(prepare_result):
        record_action(
            3, "prepare_payment", prepare_args, prepare_result
        )
        return records

    payment_id = prepare_result.get("payment_id")

    if not payment_id:
        return records

    state["payment_status"] = "PREPARED"
    record_action(
        3, "prepare_payment", prepare_args, prepare_result
    )

    # 4. Request approval.
    approval_args = {"payment_id": payment_id}

    approval_result = tools.request_approval(
        payment_id=payment_id,
        approver_id=MANAGER_ID,
        requester_id=EMPLOYEE_ID,
    )

    if not successful_action(approval_result):
        record_action(
            4, "request_approval", approval_args, approval_result
        )
        return records

    token_id = approval_result.get("approval_token_id")

    if not token_id:
        return records

    state["approval_status"] = "PENDING"
    record_action(
        4, "request_approval", approval_args, approval_result
    )

    # 5. Simulated human approval, separate from agent actions.
    human_result = tools.approve_payment(
        approval_token_id=token_id,
        approver_id=MANAGER_ID,
    )

    if successful_action(human_result):
        state["approval_status"] = "APPROVED"
    else:
        state["approval_status"] = "REJECTED"

    records.append(
        make_record(
            scenario=scenario,
            step=5,
            decision="HUMAN_APPROVAL",
            tool=None,
            arguments={},
            observation=human_result,
            workflow_state=state,
            event_type="HUMAN_APPROVAL",
        )
    )

    if not successful_action(human_result):
        return records

    # 6. Execute payment with separately authorised finance admin.
    execute_args = {
        "payment_id": payment_id,
        "approval_token_id": token_id,
    }

    execute_result = tools.execute_payment(
        payment_id=payment_id,
        approval_token_id=token_id,
        executor_id=ADMIN_ID,
    )

    if successful_action(execute_result):
        state["execution_status"] = "EXECUTED"
        state["payment_status"] = "EXECUTED"
    else:
        state["execution_status"] = "BLOCKED"

    record_action(
        6, "execute_payment", execute_args, execute_result
    )

    return records


def main():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(BENCHMARK_PATH)

    scenarios = read_jsonl(BENCHMARK_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    successful_scenarios = 0
    skipped_scenarios = 0
    failed_scenarios = 0
    errors = []

    for scenario in scenarios:
        try:
            trajectory = generate_scenario_trajectory(scenario)

            if trajectory:
                all_records.extend(trajectory)
                successful_scenarios += 1
            else:
                skipped_scenarios += 1

        except Exception as exc:
            failed_scenarios += 1
            errors.append({
                "scenario_id": scenario.get("scenario_id"),
                "error": str(exc),
            })

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    tool_counts = {}
    event_counts = {}

    for record in all_records:
        tool = record["tool"] or "NONE"
        event = record["event_type"]

        tool_counts[tool] = tool_counts.get(tool, 0) + 1
        event_counts[event] = event_counts.get(event, 0) + 1

    manifest = {
        "benchmark": "AuthorityBench",
        "split": "train",
        "input_scenarios": len(scenarios),
        "successful_scenarios": successful_scenarios,
        "skipped_scenarios": skipped_scenarios,
        "failed_scenarios": failed_scenarios,
        "trajectory_records": len(all_records),
        "tool_counts": tool_counts,
        "event_counts": event_counts,
        "errors": errors,
        "approval_policy": (
            "Simulated finance-manager approval. "
            "Approval events are separate from agent actions."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Trajectory generation completed.")
    print(json.dumps(manifest, indent=2))
    print("Trajectory file:", OUTPUT_PATH)
    print("Manifest file:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
