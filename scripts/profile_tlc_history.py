import calendar
import csv
from pathlib import Path

import pyarrow.parquet as pq


DATA_DIR = Path(r"C:\Capstone\datasets\raw\yellow_taxi")
OUTPUT_DIR = Path(r"C:\Capstone\datasets\profiles")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(filename, rows, columns):
    path = OUTPUT_DIR / filename

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated: {path}")


monthly_summary = []
schema_history = []
schema_changes = []

previous_schema = None
previous_rows = None

files = sorted(DATA_DIR.glob("yellow_tripdata_*.parquet"))

for file in files:
    month = file.stem.replace("yellow_tripdata_", "")

    year, month_number = map(int, month.split("-"))

    parquet = pq.ParquetFile(file)

    rows = parquet.metadata.num_rows

    schema = {
        field.name: str(field.type)
        for field in parquet.schema_arrow
    }

    days = calendar.monthrange(year, month_number)[1]

    file_size_mb = file.stat().st_size / (1024 * 1024)

    if previous_rows is None:
        row_change_pct = 0.0
    else:
        row_change_pct = (
            (rows - previous_rows) / previous_rows
        ) * 100

    monthly_summary.append({
        "month": month,
        "row_count": rows,
        "column_count": len(schema),
        "file_size_mb": round(file_size_mb, 2),
        "calendar_days": days,
        "average_rows_per_day": round(rows / days, 2),
        "row_change_pct": round(row_change_pct, 2),
    })

    for column_name, column_type in schema.items():
        schema_history.append({
            "month": month,
            "column_name": column_name,
            "data_type": column_type,
        })

    if previous_schema is not None:

        added = sorted(set(schema) - set(previous_schema))

        removed = sorted(set(previous_schema) - set(schema))

        type_changes = sorted(
            column
            for column in set(schema) & set(previous_schema)
            if schema[column] != previous_schema[column]
        )

        for column in added:
            schema_changes.append({
                "month": month,
                "change_type": "ADDED",
                "column_name": column,
                "previous_type": "",
                "current_type": schema[column],
            })

        for column in removed:
            schema_changes.append({
                "month": month,
                "change_type": "REMOVED",
                "column_name": column,
                "previous_type": previous_schema[column],
                "current_type": "",
            })

        for column in type_changes:
            schema_changes.append({
                "month": month,
                "change_type": "TYPE_CHANGED",
                "column_name": column,
                "previous_type": previous_schema[column],
                "current_type": schema[column],
            })

    previous_schema = schema
    previous_rows = rows

    print(
        f"{month}: {rows:,} rows, "
        f"{len(schema)} columns"
    )


save_csv(
    "monthly_summary.csv",
    monthly_summary,
    [
        "month",
        "row_count",
        "column_count",
        "file_size_mb",
        "calendar_days",
        "average_rows_per_day",
        "row_change_pct",
    ],
)

save_csv(
    "schema_history.csv",
    schema_history,
    [
        "month",
        "column_name",
        "data_type",
    ],
)

save_csv(
    "schema_changes.csv",
    schema_changes,
    [
        "month",
        "change_type",
        "column_name",
        "previous_type",
        "current_type",
    ],
)

print("\nHistorical profiling completed.")
print(f"Total months: {len(files)}")
print(f"Total records: {sum(r['row_count'] for r in monthly_summary):,}")
print(f"Schema changes found: {len(schema_changes)}")