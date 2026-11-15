from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmark_loader import load_test_scenarios
from evaluation.metrics import calculate_metrics, print_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments"


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Load JSONL predictions and reject malformed or duplicate records."""

    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    predictions: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                prediction = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in prediction file at line {line_number}"
                ) from exc

            if not isinstance(prediction, dict):
                raise ValueError(
                    f"Prediction at line {line_number} must be a JSON object."
                )

            scenario_id = prediction.get("scenario_id")

            if scenario_id is None or scenario_id == "":
                raise ValueError(
                    f"Missing scenario_id at prediction line {line_number}"
                )

            if not isinstance(scenario_id, str):
                raise ValueError(
                    f"Invalid scenario_id at prediction line {line_number}"
                )

            if scenario_id in predictions:
                raise ValueError(
                    f"Duplicate prediction for scenario: {scenario_id}"
                )

            predictions[scenario_id] = prediction

    if not predictions:
        raise ValueError("Prediction file contains no records.")

    return predictions


def build_evaluation_records(
    scenarios: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine benchmark ground truth with model predictions."""

    scenario_ids = {
        scenario["scenario_id"]
        for scenario in scenarios
    }

    prediction_ids = set(predictions)

    missing_ids = scenario_ids - prediction_ids
    unknown_ids = prediction_ids - scenario_ids

    if missing_ids:
        raise ValueError(
            f"Missing predictions for {len(missing_ids)} scenarios: "
            f"{sorted(missing_ids)[:5]}"
        )

    if unknown_ids:
        raise ValueError(
            f"Predictions contain {len(unknown_ids)} unknown scenarios: "
            f"{sorted(unknown_ids)[:5]}"
        )

    records: list[dict[str, Any]] = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        prediction = predictions[scenario_id]

        is_valid = prediction.get("valid") is True

        actual_decision = (
            prediction.get("decision")
            if is_valid
            else "INVALID"
        )

        actual_tool = (
            prediction.get("tool")
            if is_valid
            else None
        )

        actual_arguments = (
            prediction.get("arguments", {})
            if is_valid
            else {}
        )

        if not isinstance(actual_arguments, dict):
            raise ValueError(
                f"Invalid arguments for scenario {scenario_id}: "
                "expected a JSON object."
            )

        record = {
            "scenario_id": scenario_id,
            "category": scenario["category"],
            "expected_decision": scenario["expected_decision"],
            "actual_decision": actual_decision,

            "expected_tool": scenario.get("expected_tool"),
            "actual_tool": actual_tool,

            # Optional in lightweight test fixtures.
            # Present in the full AuthorityBench scenarios.
            "user_role": scenario.get("user_role"),

            "workflow_state": scenario.get("workflow_state", {}),
            "actual_arguments": actual_arguments,
            "purchase_order": scenario.get("purchase_order", {}),

            "expected_final_state": scenario.get("expected_final_state"),
            "actual_final_state": None,

            "prediction": {
                "decision": actual_decision,
                "tool": actual_tool,
                "arguments": actual_arguments,
                "reason": prediction.get("reason", ""),
                "valid": is_valid,
                "error": prediction.get("error"),
                "raw_output": prediction.get("raw_output"),
            },
        }

        records.append(record)

    return records


def calculate_extended_metrics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate decision metrics and prediction validity statistics."""

    if not records:
        raise ValueError("Cannot evaluate an empty prediction dataset.")

    metrics = calculate_metrics(records)

    valid_count = sum(
        record["prediction"]["valid"] is True
        for record in records
    )

    invalid_count = len(records) - valid_count

    metrics["valid_prediction_count"] = valid_count
    metrics["invalid_prediction_count"] = invalid_count
    metrics["valid_prediction_rate"] = valid_count / len(records)

    return metrics


def save_results(
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    model_id: str,
    output_dir: Path,
) -> Path:
    """Save metrics and per-scenario evaluation records."""

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    result = {
        "experiment": "baseline",
        "model_id": model_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(records),
        "metrics": metrics,
        "scenario_results": records,
    }

    output_path = output_dir / f"baseline_metrics_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def run_evaluation(prediction_path: Path) -> Path:
    print("Loading frozen AuthorityBench test set...")

    scenarios = load_test_scenarios()

    print(f"Loaded {len(scenarios)} benchmark scenarios.")

    print("\nLoading model predictions...")

    predictions = load_predictions(prediction_path)

    print(f"Loaded {len(predictions)} predictions.")

    print("\nBuilding evaluation records...")

    records = build_evaluation_records(
        scenarios,
        predictions,
    )

    metrics = calculate_extended_metrics(records)

    print("\n========== BASELINE EVALUATION ==========")

    print_metrics(metrics)

    model_ids = {
        prediction.get("model_id", "unknown")
        for prediction in predictions.values()
    }

    model_id = (
        next(iter(model_ids))
        if len(model_ids) == 1
        else "mixed-models"
    )

    output_path = save_results(
        metrics=metrics,
        records=records,
        model_id=model_id,
        output_dir=RESULTS_DIR / "baseline",
    )

    print(f"\nResults saved to: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SafePilot model predictions."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to baseline prediction JSONL file.",
    )

    args = parser.parse_args()

    run_evaluation(args.predictions)


if __name__ == "__main__":
    main()