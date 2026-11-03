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

    result = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,

            COUNT(*) FILTER (
                WHERE RatecodeID IS NULL
            ) AS missing_ratecode,

            COUNT(*) FILTER (
                WHERE passenger_count IS NULL
            ) AS missing_passenger,

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
        """,
        [str(file)],
    ).fetchone()

    print(f"\nMonth: {month}")
    print(f"Total rows: {result[0]:,}")
    print(f"Missing RatecodeID: {result[1]:,}")
    print(f"Missing passenger_count: {result[2]:,}")
    print(f"Both missing: {result[3]:,}")
    print(f"Only RatecodeID missing: {result[4]:,}")
    print(f"Only passenger_count missing: {result[5]:,}")

con.close()