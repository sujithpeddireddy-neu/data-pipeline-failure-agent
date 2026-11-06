"""Create a deterministic, label-balanced smoke-test from the existing benchmark.

Does not modify source CSVs. Output is appropriate for DVC tracking.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

LABELS = ("CLEAN", "F1", "F2", "F3", "F4", "F5", "F6")
COUNTS = {"train": 6, "val": 2, "test": 2}
SEED = 6170


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"Empty CSV: {path}")
        rows = list(reader)
    if headers[0] != "label" or len(headers) != 17:
        raise ValueError(f"Expected label + 16 features in {path}, got {headers}")
    if len(set(headers)) != len(headers):
        raise ValueError(f"Duplicate columns in {path}")
    for row in rows:
        if row["label"] not in LABELS:
            raise ValueError(f"Unexpected label {row['label']!r} in {path}")
        for col in headers[1:]:
            value = float(row[col])
            if not (-float('inf') < value < float('inf')):
                raise ValueError(f"Non-finite value in {path}: {col}")
    return headers, rows


def choose_f6(rows, count, rng):
    # Ensure the 2 distinct operational scenarios are represented in each split.
    missing = [r for r in rows if float(r["file_missing"]) == 1]
    late = [r for r in rows if float(r["late_arrival_flag"]) == 1]
    if not missing or not late:
        raise ValueError("F6 split lacks missing-file or late-file cases")
    selected = [rng.choice(missing), rng.choice(late)]
    if count > 2:
        remainder = [r for r in rows if r is not selected[0] and r is not selected[1]]
        selected.extend(rng.sample(remainder, count - 2))
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(r"C:\Capstone\datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("data/smoke_test"))
    args = p.parse_args()
    source = args.data_dir / "experiments" / "batch_1200_v1"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    source_hashes = {}
    output_hashes = {}
    features = None
    for split, per_label in COUNTS.items():
        source_csv = source / f"{split}.csv"
        headers, rows = read_csv(source_csv)
        if features is None:
            features = headers[1:]
        elif features != headers[1:]:
            raise ValueError(f"Feature mismatch across splits: {split}")
        groups = defaultdict(list)
        for row in rows:
            groups[row["label"]].append(row)
        rng = random.Random(SEED + list(COUNTS).index(split))
        selected = []
        for label in LABELS:
            group = groups[label]
            if len(group) < per_label:
                raise ValueError(f"Insufficient {label} cases in {split}")
            if label == "F6":
                selected.extend(choose_f6(group, per_label, rng))
            else:
                selected.extend(rng.sample(group, per_label))
        rng.shuffle(selected)
        dst = output / f"{split}.csv"
        with dst.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        source_hashes[split] = sha256(source_csv)
        output_hashes[split] = sha256(dst)
        print(f"{split}: {len(selected)} rows; " + ", ".join(f"{k}={v}" for k,v in sorted(Counter(r['label'] for r in selected).items())))
    manifest = {
        "name": "pipefail-smoke-test-v1",
        "source_dataset": "batch_1200_v1",
        "source_location_example": r"C:\Capstone\datasets\experiments\batch_1200_v1",
        "selection_method": "deterministic per-split balanced sampling without replacement; F6 includes missing and late cases",
        "seed": SEED,
        "split_counts": {s: n * len(LABELS) for s, n in COUNTS.items()},
        "samples_per_class_per_split": COUNTS,
        "feature_columns": features,
        "source_csv_sha256": source_hashes,
        "output_csv_sha256": output_hashes,
        "notes": [
            "All 16 feature columns are input features; label is the target and must not be used as input.",
            "The original 1200-case benchmark is synthetic and contains repeated feature vectors.",
            "Chronological splits are inherited from the source benchmark; this subset does not independently prove absence of leakage.",
            "Smoke test is for workflow reproducibility, not representative model-performance estimation."
        ]
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {sum(COUNTS.values()) * len(LABELS)} smoke-test rows in {output}")

if __name__ == "__main__":
    main()
