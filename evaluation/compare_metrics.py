
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / "benchmark" / "test.jsonl"
PREDICTION_DIR = ROOT / "hf_results" / "experiments" / "comparison"
OUTPUT_FILE = ROOT / "experiments" / "comparison" / "comparison_metrics.json"

VARIANTS = {
    "baseline": "baseline_predictions.jsonl",
    "standard_sft": "standard_predictions.jsonl",
    "authority_aware_sft": "authority-aware_predictions.jsonl",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    scenarios = read_jsonl(TEST_FILE)
    expected = {item["scenario_id"]: item for item in scenarios}

    if len(expected) != len(scenarios):
        raise ValueError("Duplicate scenario IDs in benchmark test set.")

    results = {}

    for variant, filename in VARIANTS.items():
        predictions = read_jsonl(PREDICTION_DIR / filename)
        prediction_map = {
            item["scenario_id"]: item for item in predictions
        }

        if len(prediction_map) != len(predictions):
            raise ValueError(f"Duplicate prediction IDs in {filename}")

        missing = set(expected) - set(prediction_map)
        extra = set(prediction_map) - set(expected)

        if missing or extra:
            raise ValueError(
                f"{variant}: missing={len(missing)}, extra={len(extra)}"
            )

        total = len(expected)
        valid_count = sum(
            p.get("valid") is True for p in predictions
        )

        decision_correct = 0
        category_stats = defaultdict(
            lambda: {"total": 0, "correct": 0, "valid": 0}
        )
        confusion = defaultdict(int)
        decision_distribution = defaultdict(int)

        for scenario_id, scenario in expected.items():
            prediction = prediction_map[scenario_id]

            expected_decision = scenario["expected_decision"]
            actual_decision = prediction.get("decision", "INVALID")
            category = scenario["category"]

            decision_distribution[actual_decision] += 1
            confusion[(expected_decision, actual_decision)] += 1

            correct = actual_decision == expected_decision
            decision_correct += int(correct)

            category_stats[category]["total"] += 1
            category_stats[category]["correct"] += int(correct)
            category_stats[category]["valid"] += int(
                prediction.get("valid") is True
            )

        category_results = {}

        for category, stats in sorted(category_stats.items()):
            category_results[category] = {
                "total": stats["total"],
                "decision_correct": stats["correct"],
                "decision_accuracy": round(
                    stats["correct"] / stats["total"], 4
                ),
                "schema_valid": stats["valid"],
                "schema_validity_rate": round(
                    stats["valid"] / stats["total"], 4
                ),
            }

        confusion_results = {
            f"{expected_label} -> {actual_label}": count
            for (expected_label, actual_label), count
            in sorted(confusion.items())
        }

        results[variant] = {
            "total": total,
            "schema_valid": valid_count,
            "schema_invalid": total - valid_count,
            "schema_validity_rate": round(valid_count / total, 4),
            "decision_correct": decision_correct,
            "decision_accuracy": round(decision_correct / total, 4),
            "decision_distribution": dict(sorted(decision_distribution.items())),
            "confusion": confusion_results,
            "per_category": category_results,
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n=== SAFEPILOT COMPARISON ===")

    for variant, metrics in results.items():
        print(f"\n{variant}")
        print(f"Schema validity: {metrics['schema_valid']}/{metrics['total']} "
              f"({metrics['schema_validity_rate']:.1%})")
        print(f"Decision accuracy: {metrics['decision_correct']}/{metrics['total']} "
              f"({metrics['decision_accuracy']:.1%})")
        print("Decision distribution:", metrics["decision_distribution"])

        print("\nPer-category decision accuracy:")
        for category, stats in metrics["per_category"].items():
            print(
                f"  {category}: "
                f"{stats['decision_correct']}/{stats['total']} "
                f"({stats['decision_accuracy']:.1%})"
            )

    print(f"\nFull metrics saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
