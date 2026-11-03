import argparse
import csv
from pathlib import Path

import duckdb


DATA_DIR = Path(r"C:\Capstone\datasets\raw\yellow_taxi")
OUTPUT_DIR = Path(r"C:\Capstone\datasets\profiles")

OUTPUT_FILE = OUTPUT_DIR / "monthly_vendor_missingness.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "month",
    "vendor_id",
    "total_records",
    "null_ratecode",
    "null_passenger_count",
    "both_missing",
    "only_ratecode_missing",
    "only_passenger_missing",
    "both_missing_pct",
]


def profile_month(conn, path):
    month = path.stem.replace("yellow_tripdata_", "")

    query = """
        SELECT
            CAST(VendorID AS VARCHAR) AS vendor_id,

            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
            ) AS null_ratecode,

            COUNT(*) FILTER (
                WHERE passenger_count IS NULL
            ) AS null_passenger_count,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
                  AND passenger_count IS NULL
            ) AS both_missing,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
                  AND passenger_count IS NOT NULL
            ) AS only_ratecode_missing,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NOT NULL
                  AND passenger_count IS NULL
            ) AS only_passenger_missing

        FROM read_parquet(?)

        GROUP BY VendorID
        ORDER BY total_records DESC
    """

    results = conn.execute(query, [str(path)]).fetchall()

    records = []

    for row in results:
        (
            vendor,
            total,
            null_ratecode,
            null_passenger,
            both,
            only_ratecode,
            only_passenger,
        ) = row

        records.append({
            "month": month,
            "vendor_id": vendor if vendor is not None else "NULL",
            "total_records": total,
            "null_ratecode": null_ratecode,
            "null_passenger_count": null_passenger,
            "both_missing": both,
            "only_ratecode_missing": only_ratecode,
            "only_passenger_missing": only_passenger,
            "both_missing_pct": round(
                100 * both / total, 4
            ) if total else 0,
        })

    return records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "months",
        nargs="*",
        help="Optional YYYY-MM months"
    )

    args = parser.parse_args()

    completed = set()

    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(newline="", encoding="utf-8") as f:
            completed = {
                row["month"]
                for row in csv.DictReader(f)
            }

    files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

    if args.months:
        files = [
            file for file in files
            if file.stem.replace("yellow_tripdata_", "")
            in args.months
        ]

    conn = duckdb.connect(database=":memory:")
    conn.execute("SET threads=4")

    write_header = (
        not OUTPUT_FILE.exists()
        or OUTPUT_FILE.stat().st_size == 0
    )

    with OUTPUT_FILE.open(
        "a", newline="", encoding="utf-8"
    ) as output:

        writer = csv.DictWriter(
            output,
            fieldnames=FIELDS
        )

        if write_header:
            writer.writeheader()

        for file in files:
            month = file.stem.replace("yellow_tripdata_", "")

            if month in completed:
                print(f"Skipping {month}")
                continue

            print(f"Profiling {month}...")

            try:
                records = profile_month(conn, file)

                writer.writerows(records)
                output.flush()

                print(
                    f"  Vendors: {len(records)} | "
                    f"Rows: {sum(r['total_records'] for r in records):,}"
                )

            except Exception as error:
                print(f"  ERROR: {error}")

    conn.close()

    print(f"\nReport: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()