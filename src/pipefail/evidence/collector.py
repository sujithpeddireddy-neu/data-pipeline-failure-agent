"""
Evidence collector for the Data Pipeline Failure Agent.

Collects:
- File availability
- Parquet schema information
- Schema differences
- Row-count differences
- DuckDB quality-check execution errors

The collector records observations, not failure labels.
"""

import argparse
import json

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

def run_revenue_probe(file_path: Path) -> dict:
    """Test whether downstream revenue aggregation succeeds."""

    con = duckdb.connect(database=":memory:")

    try:
        result = con.execute(
            """
            SELECT SUM(total_amount)
            FROM read_parquet(?)
            """,
            [str(file_path)],
        ).fetchone()

        return {
            "success": True,
            "total_revenue": result[0],
            "error_type": None,
            "error_message": None,
        }

    except duckdb.Error as error:
        return {
            "success": False,
            "total_revenue": None,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    finally:
        con.close()

def collect_null_statistics(file_path: Path) -> dict:
    """Calculate null counts and rates for every column."""

    schema = pq.read_schema(file_path)
    columns = schema.names

    expressions = []

    for index, column in enumerate(columns):
        escaped = column.replace('"', '""')

        expressions.append(
            f'COUNT_IF("{escaped}" IS NULL) AS null_{index}'
        )

    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            {", ".join(expressions)}
        FROM read_parquet(?)
    """

    con = duckdb.connect(database=":memory:")

    try:
        result = con.execute(
            query,
            [str(file_path)],
        ).fetchone()

    finally:
        con.close()

    total_rows = result[0]

    statistics = {}

    for index, column in enumerate(columns):
        null_count = int(result[index + 1] or 0)

        statistics[column] = {
            "null_count": null_count,
            "null_rate": (
                null_count / total_rows
                if total_rows > 0 else 0.0
            ),
        }

    return statistics
def collect_arrival_evidence(event_file: Path) -> dict:
    """Calculate file arrival delays from operational timestamps."""

    if not event_file.exists():
        raise FileNotFoundError(
            f"Arrival events not found: {event_file}"
        )

    events = json.loads(
        event_file.read_text(encoding="utf-8")
    )

    required_fields = [
        "expected_by_utc",
        "reference_arrived_at_utc",
        "candidate_arrived_at_utc",
        "observed_at_utc",
    ]

    timestamps = {}

    for field in required_fields:

        if field not in events:
            raise ValueError(
                f"Missing arrival timestamp: {field}"
            )

        timestamp = datetime.fromisoformat(
            events[field].replace("Z", "+00:00")
        )

        if timestamp.tzinfo is None:
            raise ValueError(
                f"Timestamp must include a timezone: {field}"
            )

        timestamps[field] = timestamp

    expected = timestamps["expected_by_utc"]
    reference_arrival = timestamps["reference_arrived_at_utc"]
    candidate_arrival = timestamps["candidate_arrived_at_utc"]
    observed = timestamps["observed_at_utc"]

    if observed < candidate_arrival:
        raise ValueError(
            "Candidate arrival occurs after the observation time."
        )

    reference_delay = (
        reference_arrival - expected
    ).total_seconds() / 60

    candidate_delay = (
        candidate_arrival - expected
    ).total_seconds() / 60

    return {
        "expected_by_utc": expected.isoformat(),
        "reference_arrived_at_utc": reference_arrival.isoformat(),
        "candidate_arrived_at_utc": candidate_arrival.isoformat(),
        "observed_at_utc": observed.isoformat(),
        "reference_delay_minutes": reference_delay,
        "candidate_delay_minutes": candidate_delay,
        "reference_late": reference_delay > 0,
        "candidate_late": candidate_delay > 0,
    }
def inspect_dataset(file_path: Path) -> dict:
    """Collect observable evidence from a Parquet dataset."""

    if not file_path.exists():
        missing_probe = {
            "success": False,
            "error_type": "FileNotFoundError",
            "error_message": "Input Parquet file is missing",
        }

        return {
            "exists": False,
            "row_count": None,
            "schema": {},
            "quality_probe": missing_probe,
            "revenue_probe": missing_probe,
            "null_statistics": {},
        }

    metadata = pq.read_metadata(file_path)
    schema = pq.read_schema(file_path)

    columns = {
        field.name: str(field.type)
        for field in schema
    }

    # Execute the same trip-distance expression
    # used by our data-quality stage.
    con = duckdb.connect(database=":memory:")

    try:
        result = con.execute(
            """
            SELECT COUNT_IF(trip_distance < 0)
            FROM read_parquet(?)
            """,
            [str(file_path)],
        ).fetchone()

        quality_probe = {
            "success": True,
            "negative_trip_distance_count": result[0],
            "error_type": None,
            "error_message": None,
        }

    except duckdb.Error as error:
        quality_probe = {
            "success": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    finally:
        con.close()

    return {
        "exists": True,
        "file_size_bytes": file_path.stat().st_size,
        "row_count": metadata.num_rows,
        "column_count": len(columns),
        "schema": columns,
        "quality_probe": quality_probe,
        "revenue_probe": run_revenue_probe(file_path),
        "null_statistics": collect_null_statistics(file_path),
    }

def collect_evidence(
    reference_file: Path,
    candidate_file: Path,
    output_file: Path,
    arrival_events_file: Path | None = None,
) -> dict:
    """Compare reference and candidate datasets."""

    reference = inspect_dataset(reference_file)
    candidate = inspect_dataset(candidate_file)

    if not reference["exists"]:
        raise FileNotFoundError(
            f"Reference dataset not found: {reference_file}"
        )

    reference_columns = reference["schema"]
    candidate_columns = candidate["schema"]

    missing_columns = sorted(
        set(reference_columns) - set(candidate_columns)
    )

    added_columns = sorted(
        set(candidate_columns) - set(reference_columns)
    )

    common_columns = (
        set(reference_columns) & set(candidate_columns)
    )

    type_changes = []

    for column in sorted(common_columns):
        original_type = reference_columns[column]
        candidate_type = candidate_columns[column]

        if original_type != candidate_type:
            type_changes.append({
                "column": column,
                "reference_type": original_type,
                "candidate_type": candidate_type,
            })
    # Missing files are availability failures.
    # Their schemas cannot be meaningfully compared.
    if not candidate["exists"]:
        missing_columns = []
        added_columns = []
        type_changes = []
    row_count_difference = None

    if candidate["row_count"] is not None:
        row_count_difference = (
            candidate["row_count"] - reference["row_count"]
        )

    # Compare null rates for columns present in both datasets.
    null_rate_changes = []

    reference_nulls = reference["null_statistics"]
    candidate_nulls = candidate["null_statistics"]

    common_null_columns = (
        set(reference_nulls) & set(candidate_nulls)
    )

    for column in sorted(common_null_columns):

        reference_rate = reference_nulls[column]["null_rate"]
        candidate_rate = candidate_nulls[column]["null_rate"]

        difference = candidate_rate - reference_rate

        if abs(difference) > 1e-9:

            null_rate_changes.append({
                "column": column,
                "reference_null_rate": reference_rate,
                "candidate_null_rate": candidate_rate,
                "delta_percentage_points": round(
                    difference * 100, 4
                ),
            })
    # Collect optional arrival-time evidence.
    arrival_evidence = None

    if arrival_events_file is not None:
        arrival_evidence = collect_arrival_evidence(
            arrival_events_file
        )
    evidence = {
        "collected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "reference": reference,
        "candidate": candidate,
        "file_availability": {
            "reference_exists": reference["exists"],
            "candidate_exists": candidate["exists"],
            "candidate_missing": not candidate["exists"],
        },
        "schema_diff": {
            "comparison_applicable": candidate["exists"],
            "missing_columns": missing_columns,
            "added_columns": added_columns,
            "type_changes": type_changes,
        },

        "row_count_difference": row_count_difference,
        "null_rate_changes": null_rate_changes,
        "arrival_evidence": arrival_evidence,
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = output_file.with_suffix(
        ".json.tmp"
    )

    temporary_file.write_text(
        json.dumps(evidence, indent=2),
        encoding="utf-8",
    )

    temporary_file.replace(output_file)

    print(
        "\nReference revenue probe:",
        "PASS" if reference["revenue_probe"]["success"]
        else "FAIL",
    )

    print(
        "Candidate revenue probe:",
        "PASS" if candidate["revenue_probe"]["success"]
        else "FAIL",
    )

    if not candidate["revenue_probe"]["success"]:
        print(
            "Candidate revenue error:",
            candidate["revenue_probe"]["error_type"],
        )

    print("\nEVIDENCE COLLECTION RESULTS")
    print("=" * 50)

    print(f"Reference rows: {reference['row_count']:,}")
    print(f"Candidate rows: {candidate['row_count']}")
    print(f"\nCandidate file exists: {candidate['exists']}")

    if not candidate["exists"]:
        print("Schema comparison skipped: candidate file missing")
    print(f"\nMissing columns: {missing_columns}")
    print(f"Added columns: {added_columns}")
    print(f"Type changes: {len(type_changes)}")

    print(
        "\nReference quality probe:",
        "PASS" if reference["quality_probe"]["success"]
        else "FAIL",
    )

    print(
        "Candidate quality probe:",
        "PASS" if candidate["quality_probe"]["success"]
        else "FAIL",
    )

    if not candidate["quality_probe"]["success"]:
        print(
            "Candidate error:",
            candidate["quality_probe"]["error_type"],
        )

    print("\nNULL-RATE CHANGES")
    print("-" * 50)

    if not null_rate_changes:
        print("No null-rate changes detected.")

    for change in null_rate_changes:

        print(
            f"{change['column']}: "
            f"{change['reference_null_rate']:.2%} -> "
            f"{change['candidate_null_rate']:.2%} "
            f"({change['delta_percentage_points']:+.2f} pp)"
        )
    if arrival_evidence is not None:

        print("\nFILE ARRIVAL EVIDENCE")
        print("-" * 50)

        print(
            "Reference delay:",
            arrival_evidence["reference_delay_minutes"],
            "minutes",
        )

        print(
            "Candidate delay:",
            arrival_evidence["candidate_delay_minutes"],
            "minutes",
        )

        print(
            "Reference late:",
            arrival_evidence["reference_late"],
        )

        print(
            "Candidate late:",
            arrival_evidence["candidate_late"],
        )
    print(f"\nEvidence saved: {output_file}")

    return evidence


def main():
    parser = argparse.ArgumentParser(
        description="Collect pipeline failure evidence."
    )

    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--arrival-events",
        required=False,
        type=Path,
        help="Optional file containing arrival timestamps",
    )
    args = parser.parse_args()

    collect_evidence(
        reference_file=args.reference,
        candidate_file=args.candidate,
        output_file=args.output,
        arrival_events_file=args.arrival_events,
    )


if __name__ == "__main__":
    main()