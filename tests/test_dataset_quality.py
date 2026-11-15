
import random

import pytest

from data_generation.generate_scenarios import build_scenario
from data_generation.audit_dataset import validate_scenario_semantics


@pytest.mark.parametrize(
    "category",
    [
        "normal_workflow",
        "approval_required",
        "indirect_prompt_injection",
        "conflicting_instructions",
        "unauthorized_action",
        "ambiguous_request",
        "approval_misuse",
    ],
)
def test_generated_scenario_has_consistent_semantics(category):
    scenario = build_scenario(
        category=category,
        index=1,
        split="train",
        rng=random.Random(42),
    )

    assert validate_scenario_semantics(scenario) == []


def test_wrong_decision_is_detected():
    scenario = build_scenario(
        category="indirect_prompt_injection",
        index=1,
        split="train",
        rng=random.Random(42),
    )

    scenario["expected_decision"] = "REFUSE"

    errors = validate_scenario_semantics(scenario)

    assert any("expected_decision" in error for error in errors)


def test_wrong_final_state_is_detected():
    scenario = build_scenario(
        category="approval_required",
        index=1,
        split="train",
        rng=random.Random(42),
    )

    scenario["expected_final_state"] = "COMPLETED"

    errors = validate_scenario_semantics(scenario)

    assert any("expected_final_state" in error for error in errors)


def test_injection_must_have_embedded_note():
    scenario = build_scenario(
        category="indirect_prompt_injection",
        index=1,
        split="train",
        rng=random.Random(42),
    )

    scenario["invoice_content"] = "Ordinary invoice content"

    errors = validate_scenario_semantics(scenario)

    assert any("embedded attack" in error for error in errors)


def test_approval_misuse_requires_expired_status():
    scenario = build_scenario(
        category="approval_misuse",
        index=1,
        split="train",
        rng=random.Random(42),
    )

    scenario["workflow_state"]["approval_status"] = "APPROVED"

    errors = validate_scenario_semantics(scenario)

    assert any("expired approval" in error for error in errors)


def test_act_tool_must_be_in_workflow_sequence():
    scenario = build_scenario(
        category="normal_workflow",
        index=1,
        split="train",
        rng=random.Random(42),
    )

    scenario["expected_tool"] = "execute_payment"

    errors = validate_scenario_semantics(scenario)

    assert any("absent from tool sequence" in error for error in errors)
