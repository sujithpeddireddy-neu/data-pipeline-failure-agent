"""
NYC TLC Yellow Taxi transformations.

Produces:
1. Daily revenue by pickup zone
2. Trip-level machine-learning features

Uses DuckDB to process Parquet files without loading
the entire dataset into a Pandas DataFrame.

Quality validation and quarantine are handled separately.
"""

import argparse
import re
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def sql_path(path: Path) -> str:
    """Escape a filesystem path for a DuckDB SQL string."""
    return path.resolve().as_posix().replace("'", "''")


def transform_month(month: str, data_dir: Path) -> dict:
    """Transform one month of NYC TLC Yellow Taxi data."""

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
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_file}"
        )
    # Require completed quarantine before downstream processing.
    accepted_file = (
        data_dir / "validated" / month / "accepted_trips.parquet"
    )

    quarantine_file = (
        data_dir / "quarantine" / month / "quarantined_trips.parquet"
    )

    if not accepted_file.exists():
        raise FileNotFoundError(
            f"Validated dataset not found: {accepted_file}"
        )

    if not quarantine_file.exists():
        raise FileNotFoundError(
            f"Quarantine dataset not found: {quarantine_file}"
        )

    # Verify that the accepted and quarantined datasets
    # account for every original record.
    raw_rows = pq.read_metadata(raw_file).num_rows
    accepted_rows = pq.read_metadata(accepted_file).num_rows
    quarantined_rows = pq.read_metadata(quarantine_file).num_rows

    if accepted_rows + quarantined_rows != raw_rows:
        raise ValueError(
            "Record reconciliation failed: "
            f"raw={raw_rows}, "
            f"accepted={accepted_rows}, "
            f"quarantined={quarantined_rows}"
        )

    output_dir = data_dir / "processed" / month
    output_dir.mkdir(parents=True, exist_ok=True)

    revenue_file = output_dir / "daily_revenue_by_zone.parquet"
    features_file = output_dir / "trip_features.parquet"

    revenue_temp = output_dir / "daily_revenue_by_zone.parquet.part"
    features_temp = output_dir / "trip_features.parquet.part"

    input_path = sql_path(accepted_file)
    revenue_path = sql_path(revenue_temp)
    features_path = sql_path(features_temp)

    source_rows = accepted_rows

    print(f"Transforming NYC TLC data: {month}")
    print(f"Input records: {source_rows:,}")

    con = duckdb.connect(database=":memory:")

    try:
        # Read the source Parquet file through DuckDB.
        con.execute(
            f"""
            CREATE VIEW trips AS
            SELECT *
            FROM read_parquet('{input_path}')
            """
        )

        # ------------------------------------------
        # OUTPUT 1: Daily revenue by pickup zone
        # ------------------------------------------

        print("Generating daily revenue by zone...")

        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(tpep_pickup_datetime AS DATE)
                        AS pickup_date,

                    PULocationID AS pickup_zone_id,

                    COUNT(*) AS trip_count,

                    SUM(total_amount) AS total_revenue

                FROM trips

                GROUP BY
                    pickup_date,
                    pickup_zone_id

                ORDER BY
                    pickup_date,
                    pickup_zone_id
            )
            TO '{revenue_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        # ------------------------------------------
        # OUTPUT 2: Trip-level ML features
        # ------------------------------------------

        print("Generating trip-level ML features...")

        con.execute(
            f"""
            COPY (
                SELECT
                    VendorID,

                    tpep_pickup_datetime,
                    tpep_dropoff_datetime,

                    PULocationID,
                    DOLocationID,

                    passenger_count,
                    RatecodeID,

                    trip_distance,
                    payment_type,

                    fare_amount,
                    tip_amount,
                    total_amount,

                    congestion_surcharge,

                    CAST(airport_fee AS DOUBLE)
                        AS airport_fee,

                    EXTRACT(
                        HOUR FROM tpep_pickup_datetime
                    ) AS pickup_hour,

                    EXTRACT(
                        DOW FROM tpep_pickup_datetime
                    ) AS pickup_day_of_week,

                    DATE_DIFF(
                        'second',
                        tpep_pickup_datetime,
                        tpep_dropoff_datetime
                    ) / 60.0 AS trip_duration_minutes

                FROM trips
            )
            TO '{features_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    except Exception:
        revenue_temp.unlink(missing_ok=True)
        features_temp.unlink(missing_ok=True)
        raise

    finally:
        con.close()

    # Verify that transformations have not silently
    # dropped any trip-level records.

    feature_rows = pq.read_metadata(features_temp).num_rows
    revenue_rows = pq.read_metadata(revenue_temp).num_rows

    if feature_rows != source_rows:
        revenue_temp.unlink(missing_ok=True)
        features_temp.unlink(missing_ok=True)

        raise ValueError(
            f"Record count mismatch: "
            f"input={source_rows}, output={feature_rows}"
        )

    # Publish completed outputs.
    revenue_temp.replace(revenue_file)
    features_temp.replace(features_file)

    results = {
        "month": month,
        "raw_rows": raw_rows,
        "accepted_rows": accepted_rows,
        "quarantined_rows": quarantined_rows,
        "source_rows": source_rows,
        "feature_rows": feature_rows,
        "revenue_rows": revenue_rows,
        "revenue_file": str(revenue_file),
        "features_file": str(features_file),
    }

    print("\nTransformation successful!")
    print(f"Original records: {raw_rows:,}")
    print(f"Quarantined records: {quarantined_rows:,}")
    print(f"Source records: {source_rows:,}")
    print(f"Feature records: {feature_rows:,}")
    print(f"Revenue groups: {revenue_rows:,}")

    print(f"\nRevenue output: {revenue_file}")
    print(f"Features output: {features_file}")

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Transform NYC TLC Yellow Taxi data."
    )

    parser.add_argument(
        "--month",
        required=True,
        help="Month in YYYY-MM format",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Root dataset directory",
    )

    args = parser.parse_args()

    transform_month(args.month, args.data_dir)


if __name__ == "__main__":
    main()