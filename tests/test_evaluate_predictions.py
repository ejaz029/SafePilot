from __future__ import annotations

from copy import deepcopy

import pytest

from evaluation.evaluate_predictions import (
    build_evaluation_records,
    calculate_extended_metrics,
    load_predictions,
)


def make_scenario(
    scenario_id: str,
    category: str = "normal_workflow",
    expected_decision: str = "ACT",
    expected_tool: str | None = "read_invoice",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "category": category,
        "expected_decision": expected_decision,
        "expected_tool": expected_tool,
        "expected_final_state": "PENDING_APPROVAL",
        "workflow_state": {
            "invoice_available": True,
            "purchase_order_available": True,
            "payment_status": "NOT_PREPARED",
            "approval_status": "NOT_REQUESTED",
            "approval_binding": "NONE",
        },
        "purchase_order": {
            "po_id": "PO-001",
            "vendor": "Acme Ltd",
            "amount": 1000,
            "currency": "USD",
            "beneficiary": "ACME-001",
        },
    }


def make_prediction(
    scenario_id: str,
    decision: str = "ACT",
    tool: str | None = "read_invoice",
    valid: bool = True,
    arguments: dict | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "model_id": "test-model",
        "decision": decision if valid else "INVALID",
        "reason": "Test prediction",
        "tool": tool if valid else None,
        "arguments": arguments or {},
        "raw_output": "{}",
        "valid": valid,
        "error": None if valid else "Invalid JSON",
    }


def test_build_evaluation_records_matches_scenarios():
    scenarios = [make_scenario("S-001"), make_scenario("S-002")]

    predictions = {
        "S-001": make_prediction("S-001"),
        "S-002": make_prediction("S-002"),
    }

    records = build_evaluation_records(scenarios, predictions)

    assert len(records) == 2
    assert records[0]["scenario_id"] == "S-001"
    assert records[1]["scenario_id"] == "S-002"


def test_missing_prediction_is_rejected():
    scenarios = [make_scenario("S-001"), make_scenario("S-002")]

    predictions = {
        "S-001": make_prediction("S-001"),
    }

    with pytest.raises(ValueError, match="Missing predictions"):
        build_evaluation_records(scenarios, predictions)


def test_unknown_prediction_is_rejected():
    scenarios = [make_scenario("S-001")]

    predictions = {
        "S-001": make_prediction("S-001"),
        "S-999": make_prediction("S-999"),
    }

    with pytest.raises(ValueError, match="unknown scenarios"):
        build_evaluation_records(scenarios, predictions)


def test_invalid_prediction_is_preserved():
    scenarios = [make_scenario("S-001")]

    predictions = {
        "S-001": make_prediction("S-001", valid=False),
    }

    records = build_evaluation_records(scenarios, predictions)

    assert records[0]["prediction"]["valid"] is False
    assert records[0]["prediction"]["decision"] == "INVALID"
    assert records[0]["actual_tool"] is None


def test_valid_prediction_preserves_tool_and_arguments():
    scenarios = [make_scenario("S-001")]

    predictions = {
        "S-001": make_prediction(
            "S-001",
            tool="prepare_payment",
            arguments={"invoice_id": "INV-001"},
        ),
    }

    records = build_evaluation_records(scenarios, predictions)

    assert records[0]["actual_tool"] == "prepare_payment"
    assert records[0]["actual_arguments"] == {
        "invoice_id": "INV-001"
    }


def test_invalid_prediction_does_not_preserve_model_tool():
    scenarios = [make_scenario("S-001")]

    invalid = make_prediction(
        "S-001",
        tool="execute_payment",
        valid=False,
    )

    invalid["tool"] = "execute_payment"

    records = build_evaluation_records(
        scenarios,
        {"S-001": invalid},
    )

    assert records[0]["actual_tool"] is None
    assert records[0]["actual_arguments"] == {}


def test_calculate_extended_metrics_includes_validity_rate():
    scenarios = [
        make_scenario("S-001"),
        make_scenario("S-002"),
    ]

    predictions = {
        "S-001": make_prediction("S-001", valid=True),
        "S-002": make_prediction("S-002", valid=False),
    }

    records = build_evaluation_records(scenarios, predictions)
    metrics = calculate_extended_metrics(records)

    assert metrics["scenario_count"] == 2
    assert metrics["valid_prediction_count"] == 1
    assert metrics["invalid_prediction_count"] == 1
    assert metrics["valid_prediction_rate"] == 0.5


def test_all_valid_predictions_have_full_validity_rate():
    scenarios = [make_scenario("S-001"), make_scenario("S-002")]

    predictions = {
        "S-001": make_prediction("S-001"),
        "S-002": make_prediction("S-002"),
    }

    records = build_evaluation_records(scenarios, predictions)
    metrics = calculate_extended_metrics(records)

    assert metrics["valid_prediction_rate"] == 1.0
    assert metrics["invalid_prediction_count"] == 0


def test_load_predictions_reads_jsonl(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"

    prediction_file.write_text(
        '{"scenario_id":"S-001","decision":"ACT"}\n'
        '{"scenario_id":"S-002","decision":"REFUSE"}\n',
        encoding="utf-8",
    )

    predictions = load_predictions(prediction_file)

    assert len(predictions) == 2
    assert predictions["S-001"]["decision"] == "ACT"
    assert predictions["S-002"]["decision"] == "REFUSE"


def test_load_predictions_rejects_duplicate_ids(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"

    prediction_file.write_text(
        '{"scenario_id":"S-001"}\n'
        '{"scenario_id":"S-001"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate prediction"):
        load_predictions(prediction_file)


def test_load_predictions_rejects_missing_scenario_id(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"

    prediction_file.write_text(
        '{"decision":"ACT"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing scenario_id"):
        load_predictions(prediction_file)


def test_load_predictions_rejects_invalid_json(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"

    prediction_file.write_text(
        '{"scenario_id": invalid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_predictions(prediction_file)


def test_load_predictions_rejects_empty_file(tmp_path):
    prediction_file = tmp_path / "predictions.jsonl"
    prediction_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no records"):
        load_predictions(prediction_file)


def test_build_records_does_not_mutate_scenarios():
    scenario = make_scenario("S-001")
    original = deepcopy(scenario)

    build_evaluation_records(
        [scenario],
        {"S-001": make_prediction("S-001")},
    )

    assert scenario == original