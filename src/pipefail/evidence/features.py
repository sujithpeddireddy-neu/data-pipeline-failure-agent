"""
Numerical feature extraction for pipeline failure evidence.

Converts evidence.json into a consistent feature vector
for failure detection and diagnosis models.

Ground-truth failure labels and experiment paths are
never included in model features.
"""

import argparse
import json
from pathlib import Path


def extract_features(evidence: dict) -> dict:
    """Extract numerical features from pipeline evidence."""

    reference = evidence["reference"]
    candidate = evidence["candidate"]

    schema_diff = evidence.get("schema_diff", {})

    candidate_exists = bool(candidate.get("exists", False))

    # ---------------------------------------
    # 1. File availability
    # ---------------------------------------

    file_missing = int(not candidate_exists)

    # ---------------------------------------
    # 2. Schema differences
    # ---------------------------------------

    schema_available = int(candidate_exists)

    missing_column_count = len(
        schema_diff.get("missing_columns", [])
    )

    added_column_count = len(
        schema_diff.get("added_columns", [])
    )

    type_change_count = len(
        schema_diff.get("type_changes", [])
    )

    # Schema comparison is not applicable
    # when the candidate file is missing.
    if not candidate_exists:
        missing_column_count = 0
        added_column_count = 0
        type_change_count = 0

    # ---------------------------------------
    # 3. Row-count changes
    # ---------------------------------------

    reference_rows = reference.get("row_count")
    candidate_rows = candidate.get("row_count")

    row_count_available = int(
        reference_rows is not None
        and candidate_rows is not None
        and reference_rows > 0
    )

    row_count_change_pct = 0.0

    if row_count_available:
        row_count_change_pct = (
            (candidate_rows - reference_rows)
            / reference_rows
        ) * 100

    # ---------------------------------------
    # 4. Null-rate changes
    # ---------------------------------------

    null_changes = evidence.get(
        "null_rate_changes", []
    )

    null_deltas = [
        change["delta_percentage_points"]
        for change in null_changes
    ]

    max_null_rate_increase_pp = max(
        [0.0] + null_deltas
    )

    max_null_rate_decrease_pp = max(
        [0.0] + [-value for value in null_deltas]
    )

    # ---------------------------------------
    # 5. Invalid-value changes
    # ---------------------------------------

    reference_probe = reference.get(
        "quality_probe", {}
    )

    candidate_probe = candidate.get(
        "quality_probe", {}
    )

    reference_invalid = reference_probe.get(
        "negative_trip_distance_count"
    )

    candidate_invalid = candidate_probe.get(
        "negative_trip_distance_count"
    )

    invalid_distance_increase_pp = 0.0

    invalid_rate_available = int(
        row_count_available
        and candidate_rows > 0
        and reference_invalid is not None
        and candidate_invalid is not None
    )

    if invalid_rate_available:
        reference_rate = (
            reference_invalid / reference_rows
        )

        candidate_rate = (
            candidate_invalid / candidate_rows
        )

        invalid_distance_increase_pp = (
            candidate_rate - reference_rate
        ) * 100

    # ---------------------------------------
    # 6. SQL execution failures
    # ---------------------------------------

    distance_probe_failed = int(
        candidate_probe.get("success") is False
    )

    revenue_probe = candidate.get(
        "revenue_probe", {}
    )

    revenue_probe_failed = int(
        revenue_probe.get("success") is False
    )

    # ---------------------------------------
    # 7. File arrival evidence
    # ---------------------------------------

    arrival = evidence.get("arrival_evidence")

    arrival_available = int(arrival is not None)

    late_arrival_minutes = 0.0
    late_arrival_flag = 0

    if arrival is not None:
        delay = arrival["candidate_delay_minutes"]

        late_arrival_minutes = max(
            0.0, float(delay)
        )

        late_arrival_flag = int(
            arrival["candidate_late"]
        )

    # ---------------------------------------
    # 8. Standard numerical feature vector
    # ---------------------------------------

    features = {
        "file_missing": file_missing,

        "schema_comparison_available": schema_available,
        "missing_column_count": missing_column_count,
        "added_column_count": added_column_count,
        "type_change_count": type_change_count,

        "row_count_available": row_count_available,
        "row_count_change_pct": round(
            row_count_change_pct, 6
        ),

        "max_null_rate_increase_pp": round(
            max_null_rate_increase_pp, 6
        ),

        "max_null_rate_decrease_pp": round(
            max_null_rate_decrease_pp, 6
        ),

        "invalid_rate_available": invalid_rate_available,

        "invalid_distance_increase_pp": round(
            invalid_distance_increase_pp, 6
        ),

        "distance_probe_failed": distance_probe_failed,
        "revenue_probe_failed": revenue_probe_failed,

        "arrival_available": arrival_available,
        "late_arrival_minutes": late_arrival_minutes,
        "late_arrival_flag": late_arrival_flag,
    }

    return features


def save_features(
    evidence_file: Path,
    output_file: Path,
) -> dict:
    """Load evidence, extract features, and save results."""

    evidence = json.loads(
        evidence_file.read_text(encoding="utf-8")
    )

    features = extract_features(evidence)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "feature_schema_version": "1.0",
        "features": features,
    }

    output_file.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nFEATURE EXTRACTION COMPLETE")
    print("=" * 45)

    for name, value in features.items():
        print(f"{name}: {value}")

    print(f"\nFeatures saved: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract numerical pipeline failure features."
    )

    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    save_features(
        evidence_file=args.evidence,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()