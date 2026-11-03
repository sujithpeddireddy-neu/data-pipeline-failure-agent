import argparse
import csv
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


DATA_DIR = Path(r"C:\Capstone\datasets\raw\yellow_taxi")
OUTPUT_DIR = Path(r"C:\Capstone\datasets\profiles")
OUTPUT_FILE = OUTPUT_DIR / "monthly_quality.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "row_count",
    "null_ratecode",
    "null_passenger_count",
    "negative_duration",
    "negative_distance",
    "negative_fare",
    "negative_total",
    "pickup_outside_month",
    "sum_total_amount",
]

FIELDS = ["month"] + METRICS + [
    "null_ratecode_pct",
    "negative_duration_pct",
    "negative_distance_pct",
]


def next_month(year, month):
    if month == 12:
        return datetime(year + 1, 1, 1)
    return datetime(year, month + 1, 1)


def profile_month(conn, file):
    month = file.stem.replace("yellow_tripdata_", "")
    year, month_num = map(int, month.split("-"))

    start = datetime(year, month_num, 1)
    end = next_month(year, month_num)

    required = {
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "RatecodeID",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "total_amount",
    }

    columns = set(pq.ParquetFile(file).schema_arrow.names)

    missing = required - columns
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    query = """
        SELECT
            COUNT(*) AS row_count,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
            ) AS null_ratecode,

            COUNT(*) FILTER (
                WHERE passenger_count IS NULL
            ) AS null_passenger_count,

            COUNT(*) FILTER (
                WHERE tpep_dropoff_datetime < tpep_pickup_datetime
            ) AS negative_duration,

            COUNT(*) FILTER (
                WHERE trip_distance < 0
            ) AS negative_distance,

            COUNT(*) FILTER (
                WHERE fare_amount < 0
            ) AS negative_fare,

            COUNT(*) FILTER (
                WHERE total_amount < 0
            ) AS negative_total,

            COUNT(*) FILTER (
                WHERE tpep_pickup_datetime < ?
                   OR tpep_pickup_datetime >= ?
            ) AS pickup_outside_month,

            ROUND(SUM(total_amount), 2) AS sum_total_amount

        FROM read_parquet(?)
    """

    values = conn.execute(
        query, [start, end, str(file)]
    ).fetchone()

    result = {"month": month}
    result.update(dict(zip(METRICS, values)))

    total = result["row_count"]

    for metric, output in [
        ("null_ratecode", "null_ratecode_pct"),
        ("negative_duration", "negative_duration_pct"),
        ("negative_distance", "negative_distance_pct"),
    ]:
        result[output] = round(
            100 * result[metric] / total, 4
        ) if total else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "months",
        nargs="*",
        help="Optional YYYY-MM months; omit to process all"
    )
    args = parser.parse_args()

    completed = set()

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(newline="", encoding="utf-8") as f:
            completed = {
                row["month"] for row in csv.DictReader(f)
            }

    files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

    if args.months:
        files = [
            f for f in files
            if f.stem.replace("yellow_tripdata_", "") in args.months
        ]

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET threads=4")

    needs_header = (
        not OUTPUT_FILE.exists()
        or OUTPUT_FILE.stat().st_size == 0
    )

    with OUTPUT_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(output, fieldnames=FIELDS)

        if needs_header:
            writer.writeheader()

        for file in files:
            month = file.stem.replace("yellow_tripdata_", "")

            if month in completed:
                print(f"Skipping {month}: already profiled")
                continue

            print(f"Profiling {month}...")

            try:
                result = profile_month(conn, file)

                writer.writerow(result)
                output.flush()

                print(
                    f"  Rows: {result['row_count']:,} | "
                    f"Null RatecodeID: {result['null_ratecode']:,} | "
                    f"Negative duration: {result['negative_duration']:,}"
                )

            except Exception as error:
                print(f"  ERROR: {error}")

    conn.close()

    print(f"\nReport: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()