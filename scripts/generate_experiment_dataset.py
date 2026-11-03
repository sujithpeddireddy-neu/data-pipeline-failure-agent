"""Build a leakage-aware, chronological synthetic failure benchmark.

Install at: scripts/generate_experiment_dataset.py
Run from the repository root with the project's activated virtual environment.

Outputs stay outside Git, under --data-dir/experiments/batch_1200_v1.
Plan: 600 train (2019-2022), 200 val (2023), 400 test (2024-Sep 2026).
Total labels: 300 CLEAN, and 150 each F1-F6. F6 splits evenly into
75 missing-file and 75 late-file cases.

The benchmark is a synthetic within-month comparison; it is not a substitute
for evaluation on naturally occurring pipeline failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURE_NAMES = (
    "file_missing",
    "schema_comparison_available",
    "missing_column_count",
    "added_column_count",
    "type_change_count",
    "row_count_available",
    "row_count_change_pct",
    "max_null_rate_increase_pp",
    "max_null_rate_decrease_pp",
    "invalid_rate_available",
    "invalid_distance_increase_pp",
    "distance_probe_failed",
    "revenue_probe_failed",
    "arrival_available",
    "late_arrival_minutes",
    "late_arrival_flag",
)

# The allocation is per failure category, not a random global row split:
# 6 * (75 + 25 + 50) + (150 + 50 + 100) = 1,200.
COUNTS = {
    "train": {"CLEAN": 150, **{f"F{i}": 75 for i in range(1, 7)}},
    "val": {"CLEAN": 50, **{f"F{i}": 25 for i in range(1, 7)}},
    "test": {"CLEAN": 100, **{f"F{i}": 50 for i in range(1, 7)}},
}

F6_MISSING_BY_SPLIT = {"train": 38, "val": 12, "test": 25}
EXPECTED_MONTHS = {"train": 48, "val": 12, "test": 33}
DEFAULT_BATCH = "batch_1200_v1"


@dataclass(frozen=True)
class Task:
    experiment_id: str
    split: str
    month: str
    label: str
    seed: int
    sample_size: int
    params: dict[str, Any]


def split_for_month(month: str) -> str | None:
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        return None
    if "2019-01" <= month <= "2022-12":
        return "train"
    if "2023-01" <= month <= "2023-12":
        return "val"
    if "2024-01" <= month <= "2026-09":
        return "test"
    return None


def available_months(data_dir: Path) -> dict[str, list[str]]:
    root = data_dir / "raw" / "yellow_taxi"
    if not root.is_dir():
        raise FileNotFoundError(f"TLC source folder not found: {root}")
    groups: dict[str, list[str]] = {key: [] for key in COUNTS}
    for file in root.glob("yellow_tripdata_????-??.parquet"):
        month = file.stem.removeprefix("yellow_tripdata_")
        split = split_for_month(month)
        if split:
            groups[split].append(month)
    for split in groups:
        groups[split] = sorted(set(groups[split]))
        if len(groups[split]) != EXPECTED_MONTHS[split]:
            raise RuntimeError(
                f"Expected {EXPECTED_MONTHS[split]} source months in {split}; "
                f"found {len(groups[split])}: {groups[split]}. "
                "Please verify the 93 monthly TLC Parquet files."
            )
    return groups


def build_plan(months: dict[str, list[str]], *, sample_size: int, plan_seed: int) -> list[Task]:
    if sample_size < 10:
        raise ValueError("Sample size must be at least 10")
    rng = random.Random(plan_seed)
    total = sum(sum(v.values()) for v in COUNTS.values())
    # Unique seeds across all incidents; no duplicated seed within a month.
    seeds = iter(rng.sample(range(1, 2_147_483_648), total))
    tasks: list[Task] = []

    for split in ("train", "val", "test"):
        labels: list[tuple[str, str]] = []
        for label, count in COUNTS[split].items():
            if label == "F6":
                n_missing = F6_MISSING_BY_SPLIT[split]
                labels.extend((label, "missing") for _ in range(n_missing))
                labels.extend((label, "late") for _ in range(count - n_missing))
            else:
                labels.extend((label, "") for _ in range(count))
        rng.shuffle(labels)

        for index, (label, scenario) in enumerate(labels, start=1):
            month = rng.choice(months[split])
            case_seed = next(seeds)
            params: dict[str, Any] = {}
            if label == "F1":
                original, renamed = rng.choice([
                    ("trip_distance", "distance"),
                    ("total_amount", "total_fare"),
                    ("passenger_count", "passengers"),
                ])
                params = {"original_column": original, "new_column": renamed}
            elif label == "F2":
                params = {"column": rng.choice([
                    "total_amount", "fare_amount", "trip_distance"
                ]), "target_type": "VARCHAR"}
            elif label == "F3":
                mode = rng.choice(["single", "correlated", "vendor"])
                params = {
                    "mode": mode,
                    "injection_fraction": round(rng.uniform(0.05, 0.40), 4),
                }
                if mode == "vendor":
                    params["vendor_id"] = 2
            elif label == "F4":
                params = {"removal_fraction": round(rng.uniform(0.05, 0.65), 4)}
            elif label == "F5":
                params = {"injection_fraction": round(rng.uniform(0.01, 0.20), 4)}
            elif label == "F6":
                params = {"scenario": scenario}
                if scenario == "late":
                    params["delay_minutes"] = rng.choice([5, 10, 15, 30, 45, 60, 90, 120])

            tasks.append(Task(
                experiment_id=f"batch60-{split}-{index:04d}",
                split=split,
                month=month,
                label=label,
                seed=case_seed,
                sample_size=sample_size,
                params=params,
            ))
    assert len(tasks) == total == 1200
    assert Counter(task.label for task in tasks)["CLEAN"] == 300
    return tasks


def load_or_create_plan(path: Path, months: dict[str, list[str]], *,
                        sample_size: int, plan_seed: int) -> list[Task]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("plan_seed") != plan_seed
                or payload.get("sample_size") != sample_size
                or payload.get("available_months") != months):
            raise ValueError(
                f"Existing plan configuration differs: {path}. "
                "Use the same arguments or a new --batch-name."
            )
        return [Task(**task) for task in payload["tasks"]]
    tasks = build_plan(months, sample_size=sample_size, plan_seed=plan_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_seed": plan_seed, "sample_size": sample_size,
        "available_months": months,
        "tasks": [asdict(t) for t in tasks],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return tasks


def target_dir(data_dir: Path, task: Task) -> Path:
    categories = {
        "CLEAN": "CLEAN",
        "F1": "F1_schema_change",
        "F2": "F2_type_change",
        "F3": "F3_null_spike",
        "F4": "F4_row_count_anomaly",
        "F5": "F5_invalid_values",
    }
    if task.label == "F6":
        folder = "F6_missing_file" if task.params["scenario"] == "missing" else "F6_late_file"
    else:
        folder = categories[task.label]
    return data_dir / "experiments" / task.month / folder / task.experiment_id


def make_clean(task: Task, data_dir: Path) -> dict[str, Any]:
    # Import at runtime so --dry-run needs no DuckDB/PyArrow installation.
    import duckdb
    import pyarrow.parquet as pq
    from pipefail.injection.failure_injector import sql_path

    folder = target_dir(data_dir, task)
    folder.mkdir(parents=True, exist_ok=True)
    baseline = folder / "baseline.parquet"
    candidate = folder / "injected.parquet"
    manifest_path = folder / "manifest.json"
    if any(p.exists() for p in (baseline, candidate, manifest_path)):
        raise FileExistsError(f"Refusing to overwrite existing clean experiment: {folder}")
    source = data_dir / "raw" / "yellow_taxi" / f"yellow_tripdata_{task.month}.parquet"
    con = duckdb.connect(database=":memory:")
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{sql_path(source)}') "
            f"USING SAMPLE {task.sample_size} ROWS (reservoir, {task.seed})) "
            f"TO '{sql_path(baseline)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()
    shutil.copyfile(baseline, candidate)
    rows = pq.read_metadata(baseline).num_rows
    if rows != task.sample_size or pq.read_metadata(candidate).num_rows != rows:
        raise ValueError(f"Unexpected clean sample row count: {folder}")
    manifest = {
        "failure_id": "CLEAN", "failure_type": "clean",
        "month": task.month, "experiment_id": task.experiment_id,
        "source_file": str(source), "baseline_file": str(baseline),
        "injected_file": str(candidate), "sample_size": rows,
        "injected_at_utc": datetime.now(timezone.utc).isoformat(),
        "injection": {"operation": "none", "random_seed": task.seed},
        "schema_preserved": True, "row_count_preserved": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def inject(task: Task, data_dir: Path) -> dict[str, Any]:
    from pipefail.injection.failure_injector import (
        inject_f1, inject_f2, inject_f3, inject_f4, inject_f5,
        inject_f6, inject_f6_late,
    )
    common = dict(month=task.month, data_dir=data_dir,
                  sample_size=task.sample_size, seed=task.seed,
                  experiment_id=task.experiment_id)
    if task.label == "CLEAN":
        return make_clean(task, data_dir)
    if task.label == "F6":
        params = dict(task.params)
        scenario = params.pop("scenario")
        return (inject_f6(**common) if scenario == "missing"
                else inject_f6_late(**common, **params))
    return {"F1": inject_f1, "F2": inject_f2, "F3": inject_f3,
            "F4": inject_f4, "F5": inject_f5}[task.label](**common, **task.params)


def load_features(features_file: Path) -> dict[str, int | float]:
    content = json.loads(features_file.read_text(encoding="utf-8"))
    if content.get("feature_schema_version") != "1.0":
        raise ValueError(f"Unexpected feature schema: {features_file}")
    values = content["features"]
    if set(values) != set(FEATURE_NAMES):
        raise ValueError(
            f"Incorrect feature columns at {features_file}: "
            f"missing={set(FEATURE_NAMES)-set(values)}, extra={set(values)-set(FEATURE_NAMES)}"
        )
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid numeric feature {name}: {value}")
    return values


def complete_case(task: Task, data_dir: Path) -> dict[str, int | float]:
    from pipefail.evidence.collector import collect_evidence
    from pipefail.evidence.features import save_features

    folder = target_dir(data_dir, task)
    manifest_file = folder / "manifest.json"
    evidence_file = folder / "evidence.json"
    feature_file = folder / "features.json"
    if feature_file.exists():
        if not manifest_file.exists():
            raise RuntimeError(f"Features exist without ground truth: {folder}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if (manifest.get("experiment_id") != task.experiment_id
                or manifest.get("failure_id") != task.label
                or manifest.get("month") != task.month):
            raise RuntimeError(f"Manifest mismatch at {folder}")
        return load_features(feature_file)
    if not manifest_file.exists():
        # No existing manifest: the case may be fresh, or partially generated.
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(
                f"Partial experiment without manifest: {folder}. "
                "Inspect the files before retrying; no automatic deletion is performed."
            )
        inject(task, data_dir)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("failure_id") != task.label or manifest.get("experiment_id") != task.experiment_id:
        raise RuntimeError(f"Manifest does not match task {task.experiment_id}")
    baseline_file = folder / "baseline.parquet"
    candidate_file = folder / "injected.parquet"
    arrival_file = folder / "arrival_events.json"
    collect_evidence(
        baseline_file,
        candidate_file,
        evidence_file,
        arrival_events_file=arrival_file if arrival_file.exists() else None,
    )
    save_features(evidence_file, feature_file)
    return load_features(feature_file)


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def export_completed(tasks: list[Task], data_dir: Path, batch_dir: Path) -> Counter:
    model_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in COUNTS}
    index_rows: list[dict[str, Any]] = []
    counts = Counter()
    for task in tasks:
        folder = target_dir(data_dir, task)
        manifest = folder / "manifest.json"
        feature_file = folder / "features.json"
        if not (manifest.exists() and feature_file.exists()):
            continue
        f = load_features(feature_file)
        model_rows[task.split].append({"label": task.label, **f})
        index_rows.append({
            "experiment_id": task.experiment_id, "split": task.split,
            "month": task.month, "label": task.label,
            "scenario": task.params.get("scenario", ""),
            "manifest_path": str(manifest), "features_path": str(feature_file),
        })
        counts[(task.split, task.label)] += 1
    # The model CSVs contain only target labels and the 16 actual evidence
    # features -- no seed, raw month, filesystem path, or injection parameters.
    for split in COUNTS:
        write_csv(batch_dir / f"{split}.csv", ["label", *FEATURE_NAMES], model_rows[split])
    write_csv(
        batch_dir / "experiment_index.csv",
        ["experiment_id", "split", "month", "label", "scenario", "manifest_path", "features_path"],
        index_rows,
    )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-name", default=DEFAULT_BATCH,
                        help="Use a new name for an independent experiment batch")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--plan-seed", type=int, default=6170)
    parser.add_argument("--dry-run", action="store_true",
                        help="Write the 1,200-case plan without touching source Parquet files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N planned cases, resumably")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.batch_name):
        parser.error("Invalid --batch-name")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.sample_size < 10:
        parser.error("--sample-size must be >= 10")
    if type(args.plan_seed) is not int:
        parser.error("--plan-seed must be an integer")

    data_dir = args.data_dir.resolve()
    months = available_months(data_dir)
    batch_dir = data_dir / "experiments" / args.batch_name
    tasks = load_or_create_plan(
        batch_dir / "plan.json", months,
        sample_size=args.sample_size, plan_seed=args.plan_seed,
    )
    print(f"Plan: {len(tasks)} cases; TRAIN=600, VAL=200, TEST=400")
    print("Labels: 300 CLEAN and 150 each F1-F6 (F6: 75 missing, 75 late)")
    print(f"Plan saved at: {batch_dir / 'plan.json'}")
    if args.dry_run:
        print("DRY RUN: No experimental Parquet files generated.")
        return

    selected = tasks if args.limit is None else tasks[:args.limit]
    for i, task in enumerate(selected, start=1):
        folder = target_dir(data_dir, task)
        if (folder / "features.json").exists():
            print(f"[{i}/{len(selected)}] SKIP completed {task.experiment_id}")
            continue
        print(f"[{i}/{len(selected)}] {task.split} {task.label} {task.month} "
              f"{task.experiment_id}", flush=True)
        try:
            complete_case(task, data_dir)
        except Exception:
            # Preserve all files for inspection. Also export completed cases.
            export_completed(tasks, data_dir, batch_dir)
            raise

    counts = export_completed(tasks, data_dir, batch_dir)
    print(f"Completed: {sum(counts.values())}/1200, by split: "
          + ", ".join(f"{s}={sum(n for (sp, _), n in counts.items() if sp == s)}"
                      for s in COUNTS))
    print(f"CSV files and index saved in: {batch_dir}")
    if len(selected) == 1200 and sum(counts.values()) == 1200:
        print("SUCCESS: Complete 1,200-experiment benchmark generated.")
    else:
        print("Partial run complete; rerun without --limit to resume all cases.")


if __name__ == "__main__":
    main()
