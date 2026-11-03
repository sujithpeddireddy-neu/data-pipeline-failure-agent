"""
Data quality checks for NYC TLC Yellow Taxi data.

Uses DuckDB for full-dataset profiling and
Great Expectations for automated validation.

All original records are preserved.
"""

import argparse
import json
import re

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import great_expectations as gx
import pandas as pd


def check_quality(month: str, data_dir: Path) -> dict:

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

    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # 1. Calculate full-dataset quality metrics
    # ----------------------------------------

    query = """
        SELECT
            COUNT(*) AS total_records,

            COUNT_IF(tpep_pickup_datetime IS NULL)
                AS null_pickup_datetime,

            COUNT_IF(tpep_dropoff_datetime IS NULL)
                AS null_dropoff_datetime,

            COUNT_IF(
                tpep_dropoff_datetime < tpep_pickup_datetime
            ) AS negative_trip_duration,

            COUNT_IF(trip_distance < 0)
                AS negative_trip_distance,

            COUNT_IF(fare_amount < 0)
                AS negative_fare_amount,

            COUNT_IF(total_amount < 0)
                AS negative_total_amount,

            COUNT_IF(total_amount IS NULL)
                AS null_total_amount,

            COUNT_IF(RatecodeID IS NULL)
                AS null_ratecode,

            COUNT_IF(passenger_count IS NULL)
                AS null_passenger_count,

            COUNT_IF(
                PULocationID IS NULL OR PULocationID <= 0
            ) AS invalid_pickup_zone,

            COUNT_IF(
                tpep_pickup_datetime < CAST(? AS TIMESTAMP)
                OR tpep_pickup_datetime >= CAST(? AS TIMESTAMP)
            ) AS pickup_outside_month

        FROM read_parquet(?)
    """

    year, month_number = map(int, month.split("-"))

    if month_number == 12:
        next_month = f"{year + 1}-01-01"
    else:
        next_month = f"{year}-{month_number + 1:02d}-01"

    start_date = f"{month}-01"

    con = duckdb.connect()

    try:
        result = con.execute(
            query,
            [start_date, next_month, str(raw_file)],
        )

        columns = [column[0] for column in result.description]
        values = result.fetchone()

        metrics = dict(zip(columns, values))

    finally:
        con.close()

    # ----------------------------------------
    # 2. Create Great Expectations batch
    # ----------------------------------------

    context = gx.get_context(mode="ephemeral")

    datasource = context.data_sources.add_pandas(
        name="tlc_quality_metrics"
    )

    asset = datasource.add_dataframe_asset(
        name="monthly_quality_profile"
    )

    batch_definition = (
        asset.add_batch_definition_whole_dataframe(
            "monthly_metrics"
        )
    )

    dataframe = pd.DataFrame([metrics])

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": dataframe}
    )

    # ----------------------------------------
    # 3. Define critical quality expectations
    # ----------------------------------------

    zero_expected = [
        "null_pickup_datetime",
        "null_dropoff_datetime",
        "negative_trip_duration",
        "negative_trip_distance",
        "null_total_amount",
        "invalid_pickup_zone",
    ]

    validation_results = {}

    for metric in zero_expected:

        expectation = (
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=metric,
                min_value=0,
                max_value=0,
            )
        )

        result = batch.validate(expectation)

        validation_results[metric] = {
            "observed_count": int(metrics[metric]),
            "passed": bool(result.success),
        }

    # Check that the dataset isn't empty.
    row_expectation = (
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_records",
            min_value=1,
        )
    )

    row_result = batch.validate(row_expectation)

    validation_results["nonempty_dataset"] = {
        "observed_count": int(metrics["total_records"]),
        "passed": bool(row_result.success),
    }

    # ----------------------------------------
    # 4. Preserve nonblocking observations
    # ----------------------------------------

    review_metrics = [
        "pickup_outside_month",
        "negative_fare_amount",
        "negative_total_amount",
        "null_ratecode",
        "null_passenger_count",
    ]

    total_records = metrics["total_records"]

    observations = {}

    for metric in review_metrics:
        count = int(metrics[metric])

        observations[metric] = {
            "count": count,
            "percentage": round(
                100 * count / total_records, 6
            ) if total_records else 0,
            "requires_review": count > 0,
        }

    failed_checks = [
        name
        for name, result in validation_results.items()
        if not result["passed"]
    ]

    # ----------------------------------------
    # 5. Generate quality report
    # ----------------------------------------

    report = {
        "dataset": "NYC TLC Yellow Taxi",
        "month": month,
        "source_file": str(raw_file),
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_records": int(total_records),
        "great_expectations": validation_results,
        "all_critical_checks_passed": not failed_checks,
        "failed_checks": failed_checks,
        "review_observations": observations,
        "quarantine_status": "not_yet_applied",
    }

    report_path = (
        metadata_dir / f"quality_report_{month}.json"
    )

    temporary_path = report_path.with_suffix(
        ".json.tmp"
    )

    temporary_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(report_path)

    # ----------------------------------------
    # 6. Display results
    # ----------------------------------------

    print(f"\nDATA QUALITY REPORT: {month}")
    print("=" * 45)

    for name, result in validation_results.items():

        status = (
            "PASS" if result["passed"] else "FAIL"
        )

        print(
            f"{name}: {status} "
            f"(count={result['observed_count']:,})"
        )

    print("\nObservations requiring review:")

    for name, observation in observations.items():

        print(
            f"{name}: {observation['count']:,} "
            f"({observation['percentage']:.4f}%)"
        )

    print(f"\nFailed critical checks: {len(failed_checks)}")
    print(f"Report saved: {report_path}")

    return report


def main():

    parser = argparse.ArgumentParser(
        description="Validate NYC TLC monthly data."
    )

    parser.add_argument(
        "--month",
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    check_quality(args.month, args.data_dir)


if __name__ == "__main__":
    main()