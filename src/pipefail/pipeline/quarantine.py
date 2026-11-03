"""
NYC TLC Yellow Taxi row quarantine.

Separates records violating critical quality rules while
preserving the original dataset and recording evidence.
"""

import argparse
import json
import re
import uuid

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def sql_path(path: Path) -> str:
    """Escape a path for a DuckDB SQL string."""
    return path.resolve().as_posix().replace("'", "''")


def quarantine_month(month: str, data_dir: Path) -> dict:
    """Separate accepted and quarantined records for one month."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    data_dir = Path(data_dir)

    raw_file = (
        data_dir
        / "raw"
        / "yellow_taxi"
        / f"yellow_tripdata_{month}.parquet"
    )

    if not raw_file.exists():
        raise FileNotFoundError(raw_file)

    accepted_dir = data_dir / "validated" / month
    quarantine_dir = data_dir / "quarantine" / month
    metadata_dir = data_dir / "metadata"

    for directory in (accepted_dir, quarantine_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    accepted_file = accepted_dir / "accepted_trips.parquet"
    quarantine_file = quarantine_dir / "quarantined_trips.parquet"

    # Unique temporary filenames prevent collisions between runs.
    run_id = uuid.uuid4().hex

    accepted_temp = accepted_dir / f"accepted_{run_id}.part"
    quarantine_temp = quarantine_dir / f"quarantined_{run_id}.part"

    source_rows = pq.read_metadata(raw_file).num_rows

    con = duckdb.connect(database=":memory:")

    try:
        # Assign quarantine reasons without changing original records.
        con.execute(
            f"""
            CREATE TEMP VIEW labeled_trips AS
            SELECT
                *,

                NULLIF(
                    CONCAT_WS(
                        ';',

                        CASE
                            WHEN tpep_pickup_datetime IS NULL
                            THEN 'NULL_PICKUP_DATETIME'
                        END,

                        CASE
                            WHEN tpep_dropoff_datetime IS NULL
                            THEN 'NULL_DROPOFF_DATETIME'
                        END,

                        CASE
                            WHEN tpep_dropoff_datetime
                                 < tpep_pickup_datetime
                            THEN 'NEGATIVE_TRIP_DURATION'
                        END,

                        CASE
                            WHEN trip_distance < 0
                            THEN 'NEGATIVE_TRIP_DISTANCE'
                        END,

                        CASE
                            WHEN total_amount IS NULL
                            THEN 'NULL_TOTAL_AMOUNT'
                        END,

                        CASE
                            WHEN PULocationID IS NULL
                                 OR PULocationID <= 0
                            THEN 'INVALID_PICKUP_ZONE'
                        END
                    ),
                    ''
                ) AS quarantine_reason

            FROM read_parquet('{sql_path(raw_file)}')
            """
        )

        print(f"\nQUARANTINE PROCESS: {month}")
        print("=" * 45)

        print("Writing accepted records...")

        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (quarantine_reason)
                FROM labeled_trips
                WHERE quarantine_reason IS NULL
            )
            TO '{sql_path(accepted_temp)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        print("Writing quarantined records...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM labeled_trips
                WHERE quarantine_reason IS NOT NULL
            )
            TO '{sql_path(quarantine_temp)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    try:
        accepted_rows = pq.read_metadata(accepted_temp).num_rows
        quarantined_rows = pq.read_metadata(quarantine_temp).num_rows

        # Critical safeguard: every original row must be accounted for.
        if accepted_rows + quarantined_rows != source_rows:
            raise ValueError(
                "Record preservation failed: "
                f"source={source_rows}, "
                f"accepted={accepted_rows}, "
                f"quarantined={quarantined_rows}"
            )

        # Publish validated outputs after count verification.
        accepted_temp.replace(accepted_file)
        quarantine_temp.replace(quarantine_file)

    finally:
        accepted_temp.unlink(missing_ok=True)
        quarantine_temp.unlink(missing_ok=True)

    # Preserve source lineage.
    ingestion_metadata_path = (
        metadata_dir / f"yellow_tripdata_{month}.json"
    )

    source_sha256 = None

    if ingestion_metadata_path.exists():
        ingestion_metadata = json.loads(
            ingestion_metadata_path.read_text(encoding="utf-8")
        )

        source_sha256 = ingestion_metadata.get("sha256")

    report = {
        "month": month,
        "run_id": run_id,
        "source_file": str(raw_file),
        "source_sha256": source_sha256,
        "source_rows": source_rows,
        "accepted_rows": accepted_rows,
        "quarantined_rows": quarantined_rows,
        "all_records_accounted_for": True,
        "accepted_file": str(accepted_file),
        "quarantine_file": str(quarantine_file),
        "original_source_preserved": True,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    report_path = (
        metadata_dir / f"quarantine_report_{month}.json"
    )

    report_temp = metadata_dir / f"quarantine_report_{run_id}.tmp"

    report_temp.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    report_temp.replace(report_path)

    print("\nQUARANTINE SUMMARY")
    print("=" * 45)
    print(f"Original records: {source_rows:,}")
    print(f"Accepted records: {accepted_rows:,}")
    print(f"Quarantined records: {quarantined_rows:,}")
    print("All records accounted for: YES")

    print(f"\nAccepted dataset: {accepted_file}")
    print(f"Quarantined dataset: {quarantine_file}")
    print(f"Report: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Quarantine invalid NYC TLC records."
    )

    parser.add_argument("--month", required=True)

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    quarantine_month(args.month, args.data_dir)


if __name__ == "__main__":
    main()