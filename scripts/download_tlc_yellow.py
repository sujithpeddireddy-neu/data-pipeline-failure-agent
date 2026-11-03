import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


DATA_DIR = Path(r"C:\Capstone\datasets\raw\yellow_taxi")

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_{}.parquet"
)

INVENTORY = DATA_DIR / "download_inventory.csv"


def get_months(start, end):
    start_date = datetime.strptime(start, "%Y-%m")
    end_date = datetime.strptime(end, "%Y-%m")

    current = start_date.year * 12 + start_date.month - 1
    final = end_date.year * 12 + end_date.month - 1

    if current > final:
        raise ValueError("Start month must not be after end month")

    while current <= final:
        year, month = divmod(current, 12)
        yield f"{year:04d}-{month + 1:02d}"
        current += 1


def validate_parquet(path):
    parquet = pq.ParquetFile(path)

    rows = parquet.metadata.num_rows
    columns = len(parquet.schema_arrow.names)

    if rows <= 0 or columns <= 0:
        raise ValueError("Empty or invalid Parquet dataset")

    return rows, columns


def record_inventory(month, status, path, rows="", columns=""):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    is_new = not INVENTORY.exists()

    with INVENTORY.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if is_new:
            writer.writerow([
                "month",
                "status",
                "rows",
                "columns",
                "size_mb",
            ])

        size_mb = (
            round(path.stat().st_size / (1024 * 1024), 2)
            if path.exists()
            else 0
        )

        writer.writerow([
            month,
            status,
            rows,
            columns,
            size_mb,
        ])


def download_month(month):
    filename = f"yellow_tripdata_{month}.parquet"

    destination = DATA_DIR / filename
    temporary = DATA_DIR / f"{filename}.part"

    url = BASE_URL.format(month)

    print(f"\nProcessing {month}")

    # Skip existing files only after validation.
    if destination.exists():
        try:
            rows, columns = validate_parquet(destination)

            print(f"Already downloaded: {rows:,} rows")
            record_inventory(
                month, "existing", destination, rows, columns
            )
            return

        except Exception as error:
            print(f"Existing file failed validation: {error}")
            print("Skipping to avoid overwriting the file.")
            record_inventory(month, "invalid_existing", destination)
            return

    command = [
        "curl.exe",
        "-4",
        "--http1.1",
        "-fL",
        "--retry", "5",
        "--retry-all-errors",
        "--retry-delay", "3",
        "--connect-timeout", "30",
        "--continue-at", "-",
        "-o", str(temporary),
        url,
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"Download failed for {month}")
        record_inventory(month, "failed", temporary)
        return

    try:
        rows, columns = validate_parquet(temporary)

        temporary.replace(destination)

        print(f"Downloaded: {rows:,} rows, {columns} columns")

        record_inventory(
            month,
            "downloaded",
            destination,
            rows,
            columns,
        )

    except Exception as error:
        print(f"Validation failed for {month}: {error}")
        record_inventory(month, "invalid", temporary)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("start_month")
    parser.add_argument("end_month")

    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for month in get_months(
        args.start_month,
        args.end_month,
    ):
        download_month(month)

    print("\nDownload batch completed.")
    print(f"Inventory: {INVENTORY}")


if __name__ == "__main__":
    main()