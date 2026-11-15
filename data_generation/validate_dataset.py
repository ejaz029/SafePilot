
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"

EXPECTED_COUNTS = {
    "train": 400,
    "validation": 100,
    "test": 150,
}


def load_jsonl(path):
    scenarios = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path.name}:{line_number}: {exc}"
                ) from exc

    return scenarios


def validate_dataset():
    all_ids = set()
    all_scenarios = {}

    for split, expected_count in EXPECTED_COUNTS.items():
        path = BENCHMARK_DIR / f"{split}.jsonl"

        if not path.exists():
            raise FileNotFoundError(path)

        scenarios = load_jsonl(path)

        if len(scenarios) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count}, "
                f"found {len(scenarios)}"
            )

        split_ids = [s["scenario_id"] for s in scenarios]

        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Duplicate scenario IDs in {split}")

        overlap = all_ids.intersection(split_ids)

        if overlap:
            raise ValueError(
                f"Scenario ID leakage between splits: {overlap}"
            )

        all_ids.update(split_ids)
        all_scenarios[split] = scenarios

        print(f"\n{split.upper()}: {len(scenarios)} scenarios")

        category_counts = Counter(
            s["category"] for s in scenarios
        )

        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")

    print("\nSPLIT INTEGRITY")
    print(f"Unique scenario IDs: {len(all_ids)}")

    for split, scenarios in all_scenarios.items():
        requests = [s["user_request"] for s in scenarios]
        unique_requests = len(set(requests))

        print(
            f"{split}: {unique_requests}/{len(requests)} "
            "unique user requests"
        )

    print("\nDATASET VALIDATION PASSED")
    print("Note: structural validation does not prove label correctness.")


if __name__ == "__main__":
    validate_dataset()
