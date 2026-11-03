import duckdb
from pathlib import Path

DATA_DIR = Path(r"C:\Capstone\datasets\raw\yellow_taxi")

MONTHS = [
    "2019-01",
    "2023-02",
    "2025-01",
    "2026-09",
]

con = duckdb.connect()

for month in MONTHS:

    file = DATA_DIR / f"yellow_tripdata_{month}.parquet"

    print(f"\n{'=' * 60}")
    print(f"MONTH: {month}")
    print(f"{'=' * 60}")

    result = con.execute(
        """
        SELECT
            VendorID,

            COUNT(*) AS total_records,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
                  AND passenger_count IS NULL
            ) AS both_missing,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE RatecodeID IS NULL
                      AND passenger_count IS NULL
                ) / COUNT(*),
                2
            ) AS missing_percentage

        FROM read_parquet(?)

        GROUP BY VendorID

        ORDER BY total_records DESC
        """,
        [str(file)],
    ).fetchall()

    print(
        f"{'VendorID':<12}"
        f"{'Total Records':<18}"
        f"{'Both Missing':<18}"
        f"{'Missing %':<12}"
    )

    for vendor, total, missing, percentage in result:
        print(
            f"{str(vendor):<12}"
            f"{total:<18,}"
            f"{missing:<18,}"
            f"{percentage:<12}"
        )

con.close()