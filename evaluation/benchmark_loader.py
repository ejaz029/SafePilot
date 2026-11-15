from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from data_generation.freeze_benchmark import verify_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

TEST_FILE = BENCHMARK_DIR / "test.jsonl"
SCHEMA_FILE = BENCHMARK_DIR / "schema.json"
MANIFEST_FILE = BENCHMARK_DIR / "manifest.json"


class BenchmarkLoadError(Exception):
    """Raised when the benchmark cannot be loaded or validated."""


def load_schema() -> dict[str, Any]:
    if not SCHEMA_FILE.exists():
        raise BenchmarkLoadError(
            f"Benchmark schema not found: {SCHEMA_FILE}"
        )

    with SCHEMA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def verify_benchmark_integrity() -> None:
    if not MANIFEST_FILE.exists():
        raise BenchmarkLoadError(
            "Benchmark manifest not found. Freeze the benchmark first."
        )

    try:
        result = verify_manifest()
    except Exception as exc:
        raise BenchmarkLoadError(
            f"Benchmark verification failed: {exc}"
        ) from exc

    # Some verification functions return None after printing success.
    # Only an explicit False indicates verification failure.
    if result is False:
        raise BenchmarkLoadError(
            "Benchmark integrity verification failed."
        )


def load_test_scenarios(
    verify_integrity: bool = True,
) -> list[dict[str, Any]]:

    if not TEST_FILE.exists():
        raise BenchmarkLoadError(
            f"Test dataset not found: {TEST_FILE}"
        )

    if verify_integrity:
        verify_benchmark_integrity()

    schema = load_schema()
    validator = Draft202012Validator(schema)

    scenarios: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with TEST_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                scenario = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkLoadError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            errors = sorted(
                validator.iter_errors(scenario),
                key=lambda error: list(error.path),
            )

            if errors:
                details = "; ".join(
                    f"{list(error.path)}: {error.message}"
                    for error in errors
                )
                raise BenchmarkLoadError(
                    f"Schema validation failed at line "
                    f"{line_number}: {details}"
                )

            scenario_id = scenario["scenario_id"]

            if scenario_id in seen_ids:
                raise BenchmarkLoadError(
                    f"Duplicate scenario ID: {scenario_id}"
                )

            seen_ids.add(scenario_id)
            scenarios.append(scenario)

    if not scenarios:
        raise BenchmarkLoadError("The test dataset is empty.")

    return scenarios


if __name__ == "__main__":
    records = load_test_scenarios()

    print(f"\nSuccessfully loaded {len(records)} test scenarios.")
    print(f"First scenario: {records[0]['scenario_id']}")