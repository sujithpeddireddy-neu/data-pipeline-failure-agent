"""Audit the finished CS6170 synthetic-failure benchmark before training.

Install: C:\\Capstone\\CS-6170\\scripts\\audit_experiment_dataset.py
Run: python scripts/audit_experiment_dataset.py --data-dir C:\\Capstone\\datasets

Read-only; no parquet files or benchmark artifacts are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from generate_experiment_dataset import COUNTS, FEATURE_NAMES, split_for_month


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--batch-name", default="batch_1200_v1")
    args = parser.parse_args()
    root = args.data_dir / "experiments" / args.batch_name
    errors: list[str] = []
    warnings: list[str] = []
    def check(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    plan_path = root / "plan.json"
    if not plan_path.exists():
        print(f"FAIL: Plan does not exist: {plan_path}")
        return 1
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = plan.get("tasks", [])
    planned_by_id = {t.get("experiment_id"): t for t in tasks}
    check(len(tasks) == 1200, f"Plan has {len(tasks)} tasks, expected 1200")
    check(len(planned_by_id) == len(tasks), "Duplicate experiment IDs in plan")

    index_headers, index = read_csv(root / "experiment_index.csv")
    expected_index_headers = ["experiment_id", "split", "month", "label", "scenario", "manifest_path", "features_path"]
    check(index_headers == expected_index_headers, f"Index headers differ: {index_headers}")
    index_by_id = {r.get("experiment_id"): r for r in index}
    check(len(index) == 1200, f"Index has {len(index)} rows, expected 1200")
    check(len(index_by_id) == len(index), "Duplicate experiment IDs in index")
    check(set(index_by_id) == set(planned_by_id), "Plan/index experiment IDs differ")

    csv_rows: dict[str, list[dict[str, str]]] = {}
    for split, desired in COUNTS.items():
        headers, rows = read_csv(root / f"{split}.csv")
        csv_rows[split] = rows
        check(headers == ["label", *FEATURE_NAMES], f"{split}.csv unexpected columns: {headers}")
        check(len(rows) == sum(desired.values()), f"{split} has {len(rows)} rows, expected {sum(desired.values())}")
        actual = Counter(r.get("label") for r in rows)
        check(actual == Counter(desired), f"{split} label counts: {dict(actual)} expected {desired}")
        for task in tasks:
            if task["split"] == split:
                check(split_for_month(task["month"]) == split, f"Chronological leakage: {task['experiment_id']} {task['month']} in {split}")

    feature_by_split_label = defaultdict(set)
    vector_to_labels = defaultdict(set)
    vector_to_splits = defaultdict(set)
    total_checked = 0
    for split, rows in csv_rows.items():
        planned = [t for t in tasks if t.get("split") == split]
        check(len(planned) == len(rows), f"{split}: plan/CSV lengths differ")
        for position, (task, row) in enumerate(zip(planned, rows), start=1):
            case_id = task["experiment_id"]
            check(row.get("label") == task["label"], f"{split} CSV row {position}: unexpected label for {case_id}")
            vals = []
            for name in FEATURE_NAMES:
                try:
                    x = float(row[name])
                    if not math.isfinite(x):
                        raise ValueError("not finite")
                    vals.append(x)
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f"{case_id}: invalid feature {name}: {exc}")
                    vals.append(float("nan"))
            vector = tuple(vals)
            if all(math.isfinite(v) for v in vals):
                vector_to_labels[vector].add(row["label"])
                vector_to_splits[vector].add(split)
                feature_by_split_label[(split, row["label"])].add(vector)

            idx = index_by_id.get(case_id)
            if idx is None:
                errors.append(f"Missing index entry: {case_id}")
                continue
            for field in ("split", "month", "label"):
                check(idx[field] == task[field], f"{case_id}: index {field} does not match plan")
            check(idx["scenario"] == task.get("params", {}).get("scenario", ""), f"{case_id}: scenario mismatch")
            mp = Path(idx["manifest_path"])
            fp = Path(idx["features_path"])
            if not mp.is_file() or not fp.is_file():
                errors.append(f"{case_id}: missing manifest or features JSON")
                continue
            try:
                manifest = json.loads(mp.read_text(encoding="utf-8"))
                content = json.loads(fp.read_text(encoding="utf-8"))
                actual_features = content["features"]
                check(content.get("feature_schema_version") == "1.0", f"{case_id}: unexpected feature schema version")
                check(set(actual_features) == set(FEATURE_NAMES), f"{case_id}: feature names mismatch in JSON")
                for name, value in zip(FEATURE_NAMES, vals):
                    if name in actual_features and math.isfinite(value):
                        check(math.isclose(float(actual_features[name]), value, rel_tol=1e-10, abs_tol=1e-10), f"{case_id}: CSV/JSON disagreement for {name}")
                check(manifest.get("experiment_id") == case_id, f"{case_id}: manifest experiment ID mismatch")
                check(manifest.get("failure_id") == task["label"], f"{case_id}: manifest label mismatch")
                check(manifest.get("month") == task["month"], f"{case_id}: manifest month mismatch")
                total_checked += 1
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"{case_id}: could not validate manifest/features JSON: {exc}")

    scenario_counts = Counter(r["scenario"] for r in index if r["label"] == "F6")
    check(scenario_counts == Counter({"missing": 75, "late": 75}), f"F6 scenarios wrong: {dict(scenario_counts)}")
    conflicting = [tuple(sorted(labels)) for labels in vector_to_labels.values() if len(labels) > 1]
    repeated_across_splits = sum(1 for splits in vector_to_splits.values() if len(splits) > 1)
    if conflicting:
        warnings.append(f"{len(conflicting)} numerical feature vectors have conflicting class labels; inspect before training")
    if repeated_across_splits:
        warnings.append(f"{repeated_across_splits} identical 16-feature vectors appear in multiple splits (possible templating; not proof of shared source records)")
    print(f"Batch: {root}")
    print(f"Indexed experiments: {len(index)}/1200 | Manifests/features verified: {total_checked}/1200")
    for split, rows in csv_rows.items():
        print(f"  {split:5}: {len(rows):4} | " + " ".join(f"{k}={Counter(r['label'] for r in rows)[k]}" for k in COUNTS[split]))
    print(f"F6 scenarios: {dict(scenario_counts)}")
    print(f"Unique feature vectors: {len(vector_to_labels)}/1200")
    for split, labels in COUNTS.items():
        print(f"  {split}: " + ", ".join(f"{label}={len(feature_by_split_label[(split, label)])} unique" for label in labels))
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL: {len(errors)} issue(s) found:")
        for issue in errors[:20]:
            print(f"  - {issue}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} additional issues")
        return 1
    print("PASS: File integrity, class counts, feature schema, and chronological split checks.")
    print("NOTE: A clean audit does not establish model performance or real-world generalization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
