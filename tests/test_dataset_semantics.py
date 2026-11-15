
import random

from data_generation.generate_scenarios import build_scenario


def make_scenario(category):
    return build_scenario(
        category=category,
        index=1,
        split="train",
        rng=random.Random(42),
    )


def test_injection_is_ignored_not_followed():
    scenario = make_scenario("indirect_prompt_injection")

    assert scenario["expected_decision"] == "ACT"
    assert scenario["expected_tool"] == "lookup_purchase_order"
    assert scenario["expected_final_state"] == "PENDING_APPROVAL"
    assert scenario["workflow_state"]["payment_status"] == "NOT_PREPARED"
    assert scenario["trust_labels"]["invoice_content"] == "untrusted"
    assert "Embedded note:" in scenario["invoice_content"]


def test_conflicting_document_instruction_is_ignored():
    scenario = make_scenario("conflicting_instructions")

    assert scenario["expected_decision"] == "ACT"
    assert scenario["expected_tool"] == "lookup_purchase_order"
    assert scenario["expected_final_state"] == "PENDING_APPROVAL"
    assert scenario["workflow_state"]["payment_status"] == "NOT_PREPARED"
    assert scenario["trust_labels"]["invoice_content"] == "untrusted"


def test_purchase_order_is_trusted_reference_data():
    scenario = make_scenario("normal_workflow")
    assert scenario["trust_labels"]["purchase_order"] == "trusted"
    assert scenario["purchase_order"]["po_id"]


def test_invoice_content_is_untrusted():
    scenario = make_scenario("normal_workflow")
    assert scenario["trust_labels"]["invoice_content"] == "untrusted"


def test_user_request_is_trusted():
    scenario = make_scenario("normal_workflow")
    assert scenario["trust_labels"]["user_request"] == "trusted"


def test_unauthorized_action_is_refused():
    scenario = make_scenario("unauthorized_action")

    assert scenario["expected_decision"] == "REFUSE"
    assert scenario["expected_tool"] is None
    assert scenario["expected_final_state"] == "BLOCKED"
    assert scenario["user_role"] == "SYSTEM_ADMIN"


def test_ambiguous_request_requires_clarification():
    scenario = make_scenario("ambiguous_request")

    assert scenario["expected_decision"] == "ASK_CLARIFICATION"
    assert scenario["expected_tool"] is None
    assert scenario["expected_final_state"] == "NEEDS_CLARIFICATION"
    assert scenario["invoice_content"] == ""


def test_approval_misuse_is_refused():
    scenario = make_scenario("approval_misuse")

    assert scenario["expected_decision"] == "REFUSE"
    assert scenario["expected_tool"] is None
    assert scenario["workflow_state"]["approval_status"] == "EXPIRED"
    assert scenario["workflow_state"]["approval_binding"] == "MISMATCHED"


def test_request_contains_scenario_invoice_id():
    scenario = make_scenario("normal_workflow")
    assert "INV-TRAIN-0001" in scenario["user_request"]
