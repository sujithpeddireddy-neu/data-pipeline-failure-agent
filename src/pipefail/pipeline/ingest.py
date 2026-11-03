"""
NYC TLC Yellow Taxi ingestion.

Downloads monthly Parquet files, validates basic file integrity,
and records ingestion metadata for reproducibility.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pyarrow.parquet as pq


TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def ingest_month(month: str, data_dir: Path) -> dict:
    """
    Ingest a monthly NYC TLC Yellow Taxi dataset.

    Args:
        month: Month in YYYY-MM format.
        data_dir: Root dataset directory.

    Returns:
        Dictionary containing ingestion metadata.
    """

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format, from 2000 onward.")

    data_dir = Path(data_dir)

    raw_dir = data_dir / "raw" / "yellow_taxi"
    metadata_dir = data_dir / "metadata"

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    filename = f"yellow_tripdata_{month}.parquet"
    source_url = f"{TLC_BASE_URL}/{filename}"

    file_path = raw_dir / filename
    metadata_path = metadata_dir / f"yellow_tripdata_{month}.json"

    downloaded = False

    # Download only if the file is not already available.
    if not file_path.exists():

        print(f"Downloading: {source_url}")

        temporary_path = raw_dir / f"{filename}.part"

        request = Request(
            source_url,
            headers={"User-Agent": "CS6170-PipeFail/0.1"},
        )

        try:
            with urlopen(request, timeout=60) as response:
                with temporary_path.open("wb") as output:

                    while True:
                        chunk = response.read(1024 * 1024)

                        if not chunk:
                            break

                        output.write(chunk)

            # Validate the downloaded Parquet file before publishing it.
            parquet_file = pq.ParquetFile(temporary_path)

            try:
                if parquet_file.metadata.num_rows == 0:
                    raise ValueError("Downloaded Parquet file contains no rows.")
            finally:
                # Windows requires the file handle to be closed before renaming.
                parquet_file.close()

            temporary_path.replace(file_path)
            downloaded = True

        finally:
            temporary_path.unlink(missing_ok=True)

    else:
        print(f"File already exists: {file_path}")

    # Read Parquet metadata without loading the entire dataset.
    parquet_file = pq.ParquetFile(file_path)

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise ValueError("Parquet file contains no rows.")

    schema = [
        {"name": field.name, "type": str(field.type)}
        for field in parquet_file.schema_arrow
    ]

    metadata = {
        "dataset": "NYC TLC Yellow Taxi",
        "month": month,
        "source_url": source_url,
        "file_path": str(file_path),
        "file_size_bytes": file_path.stat().st_size,
        "sha256": calculate_sha256(file_path),
        "row_count": row_count,
        "column_count": len(schema),
        "schema": schema,
        "downloaded_this_run": downloaded,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Save ingestion metadata atomically.
    temporary_metadata = metadata_path.with_suffix(".json.tmp")

    temporary_metadata.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    temporary_metadata.replace(metadata_path)

    print("\nIngestion successful!")
    print(f"Month: {month}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(schema)}")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"Metadata saved: {metadata_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Ingest NYC TLC monthly Yellow Taxi data."
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

    ingest_month(args.month, args.data_dir)


if __name__ == "__main__":
    main()