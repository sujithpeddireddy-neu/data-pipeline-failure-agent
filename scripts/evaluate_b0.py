"""Evaluate B0 detection and runbook diagnosis on train/val CSV data.

By default, does NOT use the locked test split. No model fitting, no repairs.
Run: python scripts/evaluate_b0.py --data-dir tests/fixtures/smoke_test
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from pipefail.detection.rules import RuleThresholds, detect_failure, load_rule_thresholds
from pipefail.diagnosis.runbook import diagnose_failure
from pipefail.contracts.schemas import FailureType


def evaluate_split(path: Path, thresholds: RuleThresholds) -> dict:
    true_positives = false_positives = false_negatives = true_negatives = 0
    correct = unknown = 0
    truth_counts: Counter[str] = Counter()
    correct_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    total = 0

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(f"Missing label column in {path}")
        for i, row in enumerate(reader, start=1):
            label = row["label"]
            if label not in {"CLEAN", "F1", "F2", "F3", "F4", "F5", "F6"}:
                raise ValueError(f"Unexpected label {label!r} in {path} row {i}")
            features = {k: float(v) for k, v in row.items() if k != "label"}
            incident_id = f"{path.stem}-{i:04d}"
            detection = detect_failure(features, incident_id, thresholds)
            diagnosis = diagnose_failure(features, incident_id, thresholds)
            predicted = (
                "CLEAN" if diagnosis.failure_type == FailureType.NONE
                else diagnosis.failure_type.value
            )
            real_positive = label != "CLEAN"
            if detection.is_suspicious and real_positive:
                true_positives += 1
            elif detection.is_suspicious:
                false_positives += 1
            elif real_positive:
                false_negatives += 1
            else:
                true_negatives += 1
            truth_counts[label] += 1
            pred_counts[predicted] += 1
            if predicted == label:
                correct += 1
                correct_counts[label] += 1
            if predicted == "UNKNOWN":
                unknown += 1
            total += 1

    if total == 0:
        raise ValueError(f"Empty split: {path}")

    def ratio(a: int, b: int) -> float | None:
        return round(a / b, 6) if b else None

    precision = ratio(true_positives, true_positives + false_positives)
    recall = ratio(true_positives, true_positives + false_negatives)
    return {
        "split": path.stem,
        "rows": total,
        "detection": {
            "tp": true_positives,
            "fp": false_positives,
            "fn": false_negatives,
            "tn": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1": (round(2 * precision * recall / (precision + recall), 6)
                   if precision is not None and recall is not None and precision + recall else None),
        },
        "diagnosis": {
            "accuracy": ratio(correct, total),
            "unknown_count": unknown,
            "per_class_recall": {
                label: ratio(correct_counts[label], truth_counts[label])
                for label in sorted(truth_counts)
            },
            "predicted_counts": dict(sorted(pred_counts.items())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B0 rules on feature CSVs")
    parser.add_argument("--data-dir", type=Path, default=Path("tests/fixtures/smoke_test"))
    parser.add_argument("--config", type=Path, default=Path("configs/b0_rules.json"))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                        default=["train", "val"])
    parser.add_argument("--output", type=Path, default=Path("results/b0_rules/metrics.json"))
    args = parser.parse_args()

    thresholds = load_rule_thresholds(args.config)
    if "test" in args.splits:
        print("WARNING: test split is reserved for a final locked evaluation.")
    result = {
        "config": str(args.config),
        "thresholds": vars(thresholds),
        "dataset": str(args.data_dir),
        "splits": [evaluate_split(args.data_dir / f"{split}.csv", thresholds)
                   for split in args.splits],
        "limitations": [
            "Evidence features are computed against a clean reference in the synthetic benchmark.",
            "Added columns alone are allowed; contract-dependent breaking changes need context.",
            "No repair execution or sandbox validation is performed by this script.",
            "Confidence is not calibrated for deterministic rules.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for entry in result["splits"]:
        det = entry["detection"]
        diag = entry["diagnosis"]
        print(
            f"{entry['split'].upper():<5} n={entry['rows']:<4} "
            f"detection_precision={det['precision']} detection_recall={det['recall']} "
            f"diagnosis_accuracy={diag['accuracy']} deferred={diag['unknown_count']}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
