
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"

DATASET_VERSION = "AuthorityBench-v1.0"
GENERATION_SEED = 20261107

SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    "train": 400,
    "validation": 100,
    "test": 150,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_scenarios(split):
    path = BENCHMARK_DIR / f"{split}.jsonl"
    scenarios = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                ) from exc

    return scenarios


def run_quality_audit():
    audit_script = ROOT / "data_generation" / "audit_dataset.py"

    print("Running dataset quality audit...")

    result = subprocess.run(
        [sys.executable, str(audit_script)],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Dataset quality audit failed. Benchmark was not frozen."
        )


def build_manifest():
    split_metadata = {}
    total_scenarios = 0

    for split in SPLITS:
        path = BENCHMARK_DIR / f"{split}.jsonl"

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        scenarios = load_scenarios(split)

        if len(scenarios) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split} has {len(scenarios)} scenarios; "
                f"expected {EXPECTED_COUNTS[split]}."
            )

        category_counts = Counter(
            scenario["category"] for scenario in scenarios
        )

        split_metadata[split] = {
            "scenario_count": len(scenarios),
            "category_counts": dict(sorted(category_counts.items())),
            "sha256": sha256_file(path),
        }

        total_scenarios += len(scenarios)

    schema_path = BENCHMARK_DIR / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError("benchmark/schema.json is missing.")

    return {
        "dataset_name": "SafePilot AuthorityBench",
        "dataset_version": DATASET_VERSION,
        "generation_seed": GENERATION_SEED,
        "total_scenarios": total_scenarios,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema": {
            "file": "schema.json",
            "sha256": sha256_file(schema_path),
        },
        "splits": split_metadata,
    }


def verify_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "No manifest found. Freeze the benchmark first."
        )

    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    errors = []

    schema_path = BENCHMARK_DIR / "schema.json"
    actual_schema_hash = sha256_file(schema_path)

    if actual_schema_hash != manifest["schema"]["sha256"]:
        errors.append("Schema checksum mismatch.")

    total = 0

    for split in SPLITS:
        path = BENCHMARK_DIR / f"{split}.jsonl"
        scenarios = load_scenarios(split)
        total += len(scenarios)

        expected = manifest["splits"][split]

        if len(scenarios) != expected["scenario_count"]:
            errors.append(
                f"{split}: scenario count changed "
                f"({len(scenarios)} vs {expected['scenario_count']})."
            )

        actual_hash = sha256_file(path)

        if actual_hash != expected["sha256"]:
            errors.append(f"{split}: SHA-256 checksum mismatch.")

        actual_categories = dict(
            sorted(Counter(
                scenario["category"] for scenario in scenarios
            ).items())
        )

        if actual_categories != expected["category_counts"]:
            errors.append(f"{split}: category distribution changed.")

    if total != manifest["total_scenarios"]:
        errors.append("Total scenario count changed.")

    if errors:
        print("\nBENCHMARK VERIFICATION FAILED")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1

    print("\nBENCHMARK VERIFICATION PASSED")
    print(f"Version: {manifest['dataset_version']}")
    print(f"Total scenarios: {manifest['total_scenarios']}")

    for split in SPLITS:
        print(
            f"{split}: "
            f"{manifest['splits'][split]['scenario_count']} scenarios "
            f"— checksum verified"
        )

    print("Schema checksum verified.")
    return 0


def freeze(refresh=False):
    if MANIFEST_PATH.exists() and not refresh:
        raise FileExistsError(
            "A manifest already exists. Use --refresh only if you "
            "intentionally want to freeze a new benchmark version."
        )

    run_quality_audit()
    manifest = build_manifest()

    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\nAUTHORITYBENCH FROZEN")
    print(f"Version: {DATASET_VERSION}")
    print(f"Seed: {GENERATION_SEED}")
    print(f"Total scenarios: {manifest['total_scenarios']}")
    print(f"Manifest: {MANIFEST_PATH}")

    for split in SPLITS:
        info = manifest["splits"][split]
        print(
            f"{split}: {info['scenario_count']} scenarios "
            f"| SHA-256: {info['sha256']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Freeze or verify the SafePilot AuthorityBench dataset."
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing benchmark files against the manifest.",
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace an existing manifest after rerunning the quality audit.",
    )

    args = parser.parse_args()

    try:
        if args.verify:
            if args.refresh:
                parser.error("--verify and --refresh cannot be used together.")

            sys.exit(verify_manifest())

        freeze(refresh=args.refresh)

    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
