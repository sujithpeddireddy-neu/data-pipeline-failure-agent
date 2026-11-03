"""
Controlled failure injection

Experiments operate on isolated dataset copies.
The original NYC TLC data is never modified.
"""
import random
import pyarrow as pa
import argparse
import json
import re
import pyarrow.compute as pc
from datetime import datetime, timezone
from pathlib import Path

import shutil
from datetime import timedelta
import duckdb
import pyarrow.parquet as pq


def sql_path(path: Path) -> str:
    """Escape a filesystem path for DuckDB SQL."""
    return path.resolve().as_posix().replace("'", "''")


def _validate_seed(seed: int) -> None:
    """DuckDB reservoir sampling accepts a nonnegative integer seed."""
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer between 0 and 2147483647.")


def _validate_experiment_id(experiment_id: str | None) -> None:
    if experiment_id is not None and not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", experiment_id
    ):
        raise ValueError("Invalid experiment ID.")


def _validate_column_name(column: str) -> None:
    if not isinstance(column, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", column
    ):
        raise ValueError(f"Invalid column name: {column}")


def inject_f1(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    seed: int = 6170,
    experiment_id: str | None = None,
    original_column: str = "trip_distance",
    new_column: str = "distance",
) -> dict:
    """Inject a configurable column-renaming failure."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

    if experiment_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment_id):
            raise ValueError("Invalid experiment ID.")

    for column in (original_column, new_column):
        _validate_column_name(column)

    if original_column.casefold() == new_column.casefold():
        raise ValueError(
            "Original and new column names must differ."
        )

    _validate_seed(seed)

    data_dir = Path(data_dir)

    source_file = (
        data_dir
        / "raw"
        / "yellow_taxi"
        / f"yellow_tripdata_{month}.parquet"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    # All experimental data remains outside the Git repository.
    experiment_dir = (
        data_dir
        / "experiments"
        / month
        / "F1_schema_change"
    )

    # Store each experiment in its own directory.
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    # Protect existing experiment results from accidental overwrites.
    if any(
        path.exists()
        for path in (baseline_file, injected_file, manifest_file)
    ):
        raise FileExistsError(
            f"Experiment files already exist in {experiment_dir}"
        )

    source_schema = pq.read_schema(source_file)

    if original_column not in source_schema.names:
        raise ValueError(
            f"Source column does not exist: {original_column}"
        )

    if new_column.lower() in {
        name.lower() for name in source_schema.names
    }:
        raise ValueError(
            f"Destination column already exists: {new_column}"
        )

    print("\nF1 — SCHEMA CHANGE INJECTION")
    print("=" * 45)
    print(f"Source month: {month}")
    print(f"Sample size: {sample_size:,}")

    con = duckdb.connect(database=":memory:")

    try:
        # Create a baseline sample from the original dataset.
        print("\nCreating unchanged reference sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        # Inject F1 into a copy of exactly the same sample.
        print("Injecting schema change...")

        con.execute(
            f"""
            COPY (
                SELECT *
                RENAME ({original_column} AS {new_column})
                FROM read_parquet('{sql_path(baseline_file)}')
            )
            TO '{sql_path(injected_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    # Verify that the injection produced the intended change.
    baseline_rows = pq.read_metadata(baseline_file).num_rows
    injected_rows = pq.read_metadata(injected_file).num_rows

    baseline_schema = pq.read_schema(baseline_file)
    injected_schema = pq.read_schema(injected_file)

    if baseline_rows != injected_rows:
        raise ValueError("F1 injection unexpectedly changed row count.")

    original_columns = baseline_schema.names
    injected_columns = injected_schema.names

    expected_columns = [
        new_column if col == original_column else col
        for col in original_columns
    ]

    if injected_columns != expected_columns:
        raise ValueError(
            "F1 schema injection verification failed."
        )

    if (
        baseline_schema.field(original_column).type
        != injected_schema.field(new_column).type
    ):
        raise ValueError(
            "F1 unexpectedly changed the column data type."
        )

    # Verify the remaining column types are unchanged.
    for old_col, new_col in zip(
        original_columns, injected_columns
    ):
        if (
            baseline_schema.field(old_col).type
            != injected_schema.field(new_col).type
        ):
            raise ValueError(
                f"Unexpected data type change: {old_col}"
            )

    # Preserve information required for later evaluation.
    manifest = {
        "failure_id": "F1",
        "failure_type": "schema_change",
        "month": month,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "experiment_id": experiment_id,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "rename_column",
            "original_column": original_column,
            "new_column": new_column,
            "random_seed": seed,
        },
        "expected_failure": "missing_expected_column",
        "row_count_preserved": True,
        "original_columns": original_columns,
        "injected_columns": injected_columns,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print(f"Original column: {original_column}")
    print(f"Modified column: {new_column}")
    print("Row count preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest

def inject_f2(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    seed: int = 6170,
    experiment_id: str | None = None,
    column: str = "total_amount",
    target_type: str = "VARCHAR",
) -> dict:
    """Change one column's data type in a reproducible sample.

    Default behavior matches the original F2 experiment:
    total_amount DOUBLE -> VARCHAR.
    """
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    _validate_seed(seed)
    _validate_experiment_id(experiment_id)
    _validate_column_name(column)

    allowed_types = {
        "VARCHAR": "string",
        "BIGINT": "int64",
        "INTEGER": "int32",
        "DOUBLE": "double",
        "FLOAT": "float",
        "BOOLEAN": "bool",
    }
    if not isinstance(target_type, str) or target_type.upper() not in allowed_types:
        raise ValueError(
            f"Target type must be one of: {', '.join(allowed_types)}"
        )
    target_type = target_type.upper()

    data_dir = Path(data_dir)
    source_file = (
        data_dir / "raw" / "yellow_taxi" / f"yellow_tripdata_{month}.parquet"
    )
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    source_schema = pq.read_schema(source_file)
    if column not in source_schema.names:
        raise ValueError(f"Source column does not exist: {column}")
    original_type = str(source_schema.field(column).type)
    if original_type == allowed_types[target_type]:
        raise ValueError(
            f"Column {column} already has the requested type {target_type}."
        )

    experiment_dir = data_dir / "experiments" / month / "F2_type_change"
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    if any(p.exists() for p in (baseline_file, injected_file, manifest_file)):
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")

    print("\nF2 — DATA TYPE CHANGE INJECTION")
    print("=" * 45)
    print(f"Source month: {month}")
    print(f"Sample size: {sample_size:,}")
    print(f"Changing {column}: {original_type} -> {target_type}")

    con = duckdb.connect(database=":memory:")
    try:
        print("\nCreating baseline sample...")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        print("Injecting data type change...")
        # Change only the requested Arrow field. A DuckDB SELECT * REPLACE
        # round-trip can also alter unrelated Parquet field nullability or
        # metadata, particularly in newer TLC monthly schemas.
        arrow_types = {
            "VARCHAR": pa.string(),
            "BIGINT": pa.int64(),
            "INTEGER": pa.int32(),
            "DOUBLE": pa.float64(),
            "FLOAT": pa.float32(),
            "BOOLEAN": pa.bool_(),
        }
        original_table = pq.read_table(baseline_file)
        column_index = original_table.schema.get_field_index(column)
        original_field = original_table.schema.field(column_index)
        try:
            cast_values = pc.cast(
                original_table.column(column), arrow_types[target_type], safe=True
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError, NotImplementedError) as exc:
            raise ValueError(
                f"Cannot convert {column} from {original_field.type} "
                f"to {target_type} without data loss: {exc}"
            ) from exc
        converted_table = original_table.set_column(
            column_index,
            original_field.with_type(arrow_types[target_type]),
            cast_values,
        )
        pq.write_table(converted_table, injected_file, compression="zstd")
    except Exception:
        baseline_file.unlink(missing_ok=True)
        injected_file.unlink(missing_ok=True)
        raise
    finally:
        con.close()

    baseline_schema = pq.read_schema(baseline_file)
    injected_schema = pq.read_schema(injected_file)
    baseline_rows = pq.read_metadata(baseline_file).num_rows
    injected_rows = pq.read_metadata(injected_file).num_rows

    if baseline_rows != injected_rows:
        raise ValueError("F2 unexpectedly changed row count.")
    if baseline_schema.names != injected_schema.names:
        raise ValueError("F2 unexpectedly changed column names.")

    original_type = str(baseline_schema.field(column).type)
    modified_type = str(injected_schema.field(column).type)
    if modified_type != allowed_types[target_type] or original_type == modified_type:
        raise ValueError(
            f"Unexpected F2 types for {column}: {original_type} -> {modified_type}"
        )
    # An F2 incident must change just one column's type, never the other
    # fields, their nullability, metadata, or values.
    baseline_table = pq.read_table(baseline_file)
    injected_table = pq.read_table(injected_file)
    if baseline_table[column].null_count != injected_table[column].null_count:
        raise ValueError(f"F2 unexpectedly changed null count of {column}")
    for other_column in baseline_schema.names:
        if other_column == column:
            continue
        if baseline_schema.field(other_column) != injected_schema.field(other_column):
            raise ValueError(f"F2 unexpectedly changed schema for {other_column}")
        if not baseline_table[other_column].equals(injected_table[other_column]):
            raise ValueError(f"F2 unexpectedly changed values of {other_column}")

    manifest = {
        "failure_id": "F2",
        "failure_type": "data_type_change",
        "month": month,
        "experiment_id": experiment_id,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(timezone.utc).isoformat(),
        "injection": {
            "operation": "change_column_type",
            "column": column,
            "original_type": original_type,
            "modified_type": modified_type,
            "target_type": target_type,
            "random_seed": seed,
        },
        "expected_failure": (
            "incompatible_numeric_aggregation"
            if column == "total_amount" and modified_type == "string"
            else "unexpected_column_type"
        ),
        "row_count_preserved": True,
        "column_names_preserved": True,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print(f"Modified column: {column}")
    print(f"Type change: {original_type} -> {modified_type}")
    print("Row count preserved: YES")
    print(f"\nManifest: {manifest_file}")
    return manifest


def inject_f3(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    injection_fraction: float = 0.30,
    seed: int = 6170,
    experiment_id: str | None = None,
    mode: str = "single",
    vendor_id: int | None = None,
) -> dict:
    """Inject configurable missing-value failures into sampled data."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

    if not 0 < injection_fraction <= 1:
        raise ValueError(
            "Injection fraction must be greater than 0 and at most 1."
        )

    if mode not in ("single", "correlated", "vendor"):
        raise ValueError(
            "Mode must be single, correlated, or vendor."
        )

    if mode == "vendor" and vendor_id is None:
        raise ValueError(
            "Vendor ID is required for vendor-specific injection."
        )

    if mode != "vendor" and vendor_id is not None:
        raise ValueError(
            "Vendor ID can only be specified in vendor mode."
        )

    if experiment_id is not None:
        if not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}",
            experiment_id,
        ):
            raise ValueError("Invalid experiment ID.")

    _validate_seed(seed)

    data_dir = Path(data_dir)

    source_file = (
        data_dir
        / "raw"
        / "yellow_taxi"
        / f"yellow_tripdata_{month}.parquet"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    experiment_dir = (
        data_dir
        / "experiments"
        / month
        / "F3_null_spike"
    )

    # Keep each experiment in its own directory.
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id

    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    if any(
        p.exists()
        for p in (baseline_file, injected_file, manifest_file)
    ):
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    print("\nF3 — NULL-RATE SPIKE INJECTION")
    print("=" * 45)

    con = duckdb.connect(database=":memory:")

    try:
        print("Creating baseline sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    # Load only the sampled baseline, not the full monthly dataset.
    baseline_table = pq.read_table(baseline_file)
    baseline_rows = baseline_table.num_rows

    if baseline_rows < 10:
        raise ValueError("Insufficient records for F3 injection.")

    # Determine which columns will receive additional null values.
    if mode == "single":
        target_columns = ["RatecodeID"]
    else:
        target_columns = ["RatecodeID", "passenger_count"]

    required_columns = set(target_columns)

    if mode == "vendor":
        required_columns.add("VendorID")

    missing_columns = (
        required_columns - set(baseline_table.column_names)
    )

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    ratecodes = baseline_table["RatecodeID"].to_pylist()

    passengers = (
        baseline_table["passenger_count"].to_pylist()
        if mode != "single"
        else None
    )

    vendors = (
        baseline_table["VendorID"].to_pylist()
        if mode == "vendor"
        else None
    )

    # Select records where introducing nulls is possible.
    eligible_indices = []

    for index in range(baseline_rows):

        if ratecodes[index] is None:
            continue

        if mode != "single" and passengers[index] is None:
            continue

        if mode == "vendor" and vendors[index] != vendor_id:
            continue

        eligible_indices.append(index)

    if not eligible_indices:
        raise ValueError(
            "No eligible records found for F3 injection."
        )

    # Calculate how many eligible records to modify.
    target_count = max(
        1,
        int(len(eligible_indices) * injection_fraction)
    )

    # Select records reproducibly.
    rng = random.Random(seed)

    selected_indices = set(
        rng.sample(eligible_indices, target_count)
    )

    selection_mask = pa.array(
        [
            index in selected_indices
            for index in range(baseline_rows)
        ],
        type=pa.bool_(),
    )

    print(
        f"Injecting F3: mode={mode}, "
        f"fraction={injection_fraction:.2%}, "
        f"records={target_count:,}"
    )

    injected_table = baseline_table

    # Modify only the selected columns and rows.
    for column in target_columns:

        column_index = baseline_table.schema.get_field_index(
            column
        )

        original_field = baseline_table.schema.field(
            column_index
        )

        modified_values = pc.if_else(
            selection_mask,
            pa.scalar(None, type=original_field.type),
            baseline_table[column],
        )

        injected_table = injected_table.set_column(
            column_index,
            original_field,
            modified_values,
        )

    # Save the injected dataset.
    pq.write_table(
        injected_table,
        injected_file,
        compression="zstd",
    )

    # Verify the generated dataset.
    injected_check = pq.read_table(injected_file)

    if injected_check.num_rows != baseline_rows:
        raise ValueError("F3 unexpectedly changed row count.")

    if pq.read_schema(baseline_file) != pq.read_schema(
        injected_file
    ):
        raise ValueError("F3 unexpectedly changed the schema.")

    # Every targeted column must gain exactly target_count nulls.
    for column in target_columns:

        original_nulls = baseline_table[column].null_count
        new_nulls = injected_check[column].null_count

        if new_nulls - original_nulls != target_count:
            raise ValueError(
                f"F3 injection verification failed for {column}."
            )

    # All other columns must remain unchanged.
    for column in baseline_table.column_names:

        if column in target_columns:
            continue

        if not baseline_table[column].equals(
            injected_check[column]
        ):
            raise ValueError(
                f"F3 unexpectedly modified column {column}."
            )

    baseline_nulls = baseline_table["RatecodeID"].null_count
    injected_nulls = injected_check["RatecodeID"].null_count

    baseline_rate = baseline_nulls / baseline_rows
    injected_rate = injected_nulls / baseline_rows

    print("\nF3 INJECTION VERIFIED")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Eligible records: {len(eligible_indices):,}")
    print(f"Modified records: {target_count:,}")
    print(f"Baseline null rate: {baseline_rate:.2%}")
    print(f"Injected null rate: {injected_rate:.2%}")
    print(f"Schema preserved: YES")
    print(f"Row count preserved: YES")

    # Record the actual failure parameters as ground truth.
    # These labels must NOT become AI input features.
    manifest = {
        "failure_id": "F3",
        "failure_type": "null_rate_spike",
        "month": month,
        "experiment_id": experiment_id,

        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),

        "sample_size": baseline_rows,

        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "injection": {
            "operation": "introduce_null_values",
            "mode": mode,
            "target_columns": target_columns,
            "vendor_id": vendor_id,
            "injection_fraction": injection_fraction,
            "eligible_record_count": len(eligible_indices),
            "modified_records": target_count,
            "random_seed": seed,

            "baseline_null_rate": baseline_rate,
            "observed_null_rate": injected_rate,
            "null_rate_increase_pp": round(
                (injected_rate - baseline_rate) * 100,
                4
            ),
        },

        "baseline_null_count": baseline_nulls,
        "injected_null_count": injected_nulls,

        "row_count_preserved": True,
        "schema_preserved": True,
        "other_columns_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_check.num_rows:,}")
    print(f"Modified columns: {', '.join(target_columns)}")
    print(f"Baseline nulls: {baseline_nulls:,}")
    print(f"Injected nulls: {injected_nulls:,}")
    print(f"Baseline null rate: {baseline_rate:.2%}")
    print(f"Injected null rate: {injected_rate:.2%}")
    print("Schema preserved: YES")
    print("Row count preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest

def inject_f4(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    removal_fraction: float = 0.40,
    seed: int = 6170,
    experiment_id: str | None = None,
) -> dict:
    """Inject a configurable row-count reduction."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

    # Validate removal percentage
    if not 0 < removal_fraction < 1:
        raise ValueError("Removal fraction must be between 0 and 1.")

    # Validate experiment ID
    if experiment_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment_id):
            raise ValueError("Invalid experiment ID.")

    _validate_seed(seed)

    data_dir = Path(data_dir)

    source_file = (
        data_dir
        / "raw"
        / "yellow_taxi"
        / f"yellow_tripdata_{month}.parquet"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    experiment_dir = (
        data_dir
        / "experiments"
        / month
        / "F4_row_count_anomaly"
    )
    # Create a separate folder for each experiment
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id

    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    if any(
        path.exists()
        for path in (baseline_file, injected_file, manifest_file)
    ):
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    print("\nF4 — ROW-COUNT ANOMALY INJECTION")
    print("=" * 45)

    # Create an unchanged baseline sample.
    con = duckdb.connect(database=":memory:")

    try:
        print("Creating baseline sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    # Load only the sampled records, not the full dataset.
    baseline_table = pq.read_table(baseline_file)

    baseline_rows = baseline_table.num_rows

    if baseline_rows < 10:
        raise ValueError("Insufficient records for F4 injection.")

    # Remove the configured percentage of records.
    removed_rows = int(baseline_rows * removal_fraction)
    retained_rows = baseline_rows - removed_rows

    # Use the provided seed for reproducible row selection.
    rng = random.Random(seed)

    retained_indices = sorted(
        rng.sample(range(baseline_rows), retained_rows)
    )

    print("Injecting row-count anomaly...")

    # Arrow's take() preserves the original column schema.
    injected_table = baseline_table.take(
        pa.array(retained_indices, type=pa.int64())
    )

    pq.write_table(
        injected_table,
        injected_file,
        compression="zstd",
    )

    # Verify actual record counts.
    actual_baseline_rows = pq.read_metadata(
        baseline_file
    ).num_rows

    actual_injected_rows = pq.read_metadata(
        injected_file
    ).num_rows

    if actual_baseline_rows != baseline_rows:
        raise ValueError("Baseline row count changed unexpectedly.")

    if actual_injected_rows != retained_rows:
        raise ValueError("F4 row-count injection failed.")

    # Verify column names, types, and nullability.
    baseline_schema = pq.read_schema(baseline_file)
    injected_schema = pq.read_schema(injected_file)

    if baseline_schema.names != injected_schema.names:
        raise ValueError("F4 unexpectedly changed column names.")

    for original, modified in zip(
        baseline_schema,
        injected_schema,
    ):
        if original != modified:
            raise ValueError(
                f"F4 unexpectedly changed schema field: {original.name}"
            )

    observed_reduction = (
        baseline_rows - actual_injected_rows
    ) / baseline_rows

    manifest = {
        "failure_id": "F4",
        "failure_type": "row_count_anomaly",
        "month": month,
        "experiment_id": experiment_id,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "remove_rows",
            "target_removal_fraction": removal_fraction,
            "observed_removal_fraction": observed_reduction,
            "random_seed": seed,
        },
        "baseline_row_count": baseline_rows,
        "injected_row_count": actual_injected_rows,
        "removed_row_count": removed_rows,
        "schema_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {actual_injected_rows:,}")
    print(f"Removed records: {removed_rows:,}")
    print(f"Row-count reduction: {observed_reduction:.2%}")
    print("Schema preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest

def inject_f5(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    injection_fraction: float = 0.05,
    seed: int = 6170,
    experiment_id: str | None = None,
) -> dict:
    """Inject configurable negative trip distances into sampled data."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

    if not 0 < injection_fraction <= 1:
        raise ValueError(
            "Injection fraction must be greater than 0 and at most 1."
        )

    if experiment_id is not None:
        if not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}",
            experiment_id,
        ):
            raise ValueError("Invalid experiment ID.")

    _validate_seed(seed)

    data_dir = Path(data_dir)

    source_file = (
        data_dir
        / "raw"
        / "yellow_taxi"
        / f"yellow_tripdata_{month}.parquet"
    )

    if not source_file.exists():
        raise FileNotFoundError(source_file)

    experiment_dir = (
        data_dir
        / "experiments"
        / month
        / "F5_invalid_values"
    )

    # Store each experiment in a separate directory.
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id

    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    if any(
        p.exists()
        for p in (baseline_file, injected_file, manifest_file)
    ):
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    print("\nF5 — INVALID VALUE INJECTION")
    print("=" * 45)

    # Step 1: Generate the baseline sample.
    con = duckdb.connect(database=":memory:")

    try:
        print("Creating baseline sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    # Step 2: Load the sampled records.
    baseline_table = pq.read_table(baseline_file)

    baseline_rows = baseline_table.num_rows

    column_index = baseline_table.schema.get_field_index(
        "trip_distance"
    )

    if column_index < 0:
        raise ValueError("trip_distance column not found.")

    column_field = baseline_table.schema.field(column_index)

    if not pa.types.is_floating(column_field.type):
        raise ValueError("trip_distance must be numeric.")

    distances = baseline_table.column(
        "trip_distance"
    ).to_pylist()

    # Only select records that do not already have
    # a negative distance.
    eligible_indices = [
        index
        for index, distance in enumerate(distances)
        if distance is not None and distance >= 0
    ]

    # Ensure valid records are available for injection.
    if not eligible_indices:
        raise ValueError(
            "No eligible records available for F5 injection."
        )

    # Calculate severity using only eligible records.
    target_count = max(
        1,
        int(len(eligible_indices) * injection_fraction)
    )

    # Select records using the configured random seed.
    rng = random.Random(seed)

    selected_indices = set(
        rng.sample(eligible_indices, target_count)
    )

    mask = pa.array(
        [
            index in selected_indices
            for index in range(baseline_rows)
        ],
        type=pa.bool_(),
    )

    print("Injecting negative trip distances...")

    # Change only selected trip_distance values.
    modified_distances = pc.if_else(
        mask,
        pa.scalar(-1.0, type=column_field.type),
        baseline_table.column("trip_distance"),
    )

    injected_table = baseline_table.set_column(
        column_index,
        column_field,
        modified_distances,
    )

    # PyArrow preserves the other fields and their
    # original schema definitions.
    pq.write_table(
        injected_table,
        injected_file,
        compression="zstd",
    )

    # Step 4: Verify that injection was controlled.
    baseline_schema = pq.read_schema(baseline_file)
    injected_schema = pq.read_schema(injected_file)

    if baseline_schema != injected_schema:
        raise ValueError("F5 unexpectedly changed the schema.")

    injected_check = pq.read_table(injected_file)

    if injected_check.num_rows != baseline_rows:
        raise ValueError("F5 unexpectedly changed row count.")

    # Verify all other columns remain unchanged.
    for column in baseline_schema.names:
        if column == "trip_distance":
            continue

        if not baseline_table[column].equals(
            injected_check[column]
        ):
            raise ValueError(
                f"F5 unexpectedly modified {column}"
            )

    original_distances = distances

    new_distances = injected_check.column(
        "trip_distance"
    ).to_pylist()

    baseline_invalid = sum(
        value is not None and value < 0
        for value in original_distances
    )

    injected_invalid = sum(
        value is not None and value < 0
        for value in new_distances
    )

    if injected_invalid - baseline_invalid != target_count:
        raise ValueError(
            "F5 invalid-value count verification failed."
        )

    changed_count = sum(
        old != new
        for old, new in zip(
            original_distances,
            new_distances,
        )
    )

    if changed_count != target_count:
        raise ValueError(
            "F5 modified an unexpected number of values."
        )

    # Step 5: Record ground truth.
    manifest = {
        "failure_id": "F5",
        "failure_type": "invalid_values",
        "month": month,
        "experiment_id": experiment_id,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "replace_values",
            "column": "trip_distance",
            "injected_value": -1.0,

            # Configured experiment parameters.
            "target_fraction": injection_fraction,
            "random_seed": seed,

            # Actual injection results.
            "eligible_record_count": len(eligible_indices),
            "modified_records": changed_count,
            "observed_fraction_of_eligible": (
                changed_count / len(eligible_indices)
            ),
            "observed_fraction_of_total": (
                changed_count / baseline_rows
            ),
        },
        "baseline_invalid_count": baseline_invalid,
        "injected_invalid_count": injected_invalid,
        "schema_preserved": True,
        "row_count_preserved": True,
        "other_columns_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_check.num_rows:,}")
    print(f"Modified records: {changed_count:,}")
    print(f"Baseline invalid distances: {baseline_invalid:,}")
    print(f"Injected invalid distances: {injected_invalid:,}")
    print(f"Injected fraction: {changed_count / baseline_rows:.2%}")
    print("Schema preserved: YES")
    print("Row count preserved: YES")
    print("Other columns preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest

def inject_f6(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    seed: int = 6170,
    experiment_id: str | None = None,
) -> dict:
    """Simulate a missing candidate file without removing the original TLC data."""
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    _validate_seed(seed)
    _validate_experiment_id(experiment_id)

    data_dir = Path(data_dir)
    source_file = (
        data_dir / "raw" / "yellow_taxi" / f"yellow_tripdata_{month}.parquet"
    )
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    experiment_dir = data_dir / "experiments" / month / "F6_missing_file"
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"
    if any(p.exists() for p in (baseline_file, injected_file, manifest_file)):
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")

    print("\nF6 — MISSING FILE INJECTION")
    print("=" * 45)
    con = duckdb.connect(database=":memory:")
    try:
        print("Creating baseline sample...")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    baseline_rows = pq.read_metadata(baseline_file).num_rows
    if baseline_rows == 0:
        raise ValueError("Baseline dataset is empty.")

    print("Simulating missing expected input file...")
    if injected_file.exists():
        raise ValueError("F6 failed: candidate file unexpectedly exists.")

    check_time = datetime.now(timezone.utc).isoformat()
    manifest = {
        "failure_id": "F6",
        "failure_type": "missing_or_late_file",
        "scenario": "missing_file",
        "month": month,
        "experiment_id": experiment_id,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": check_time,
        "injection": {
            "operation": "simulate_missing_file",
            "expected_path": str(injected_file),
            "file_exists": False,
            "check_time_utc": check_time,
            "random_seed": seed,
        },
        "expected_failure": "file_not_found",
        "baseline_exists": True,
        "candidate_exists": False,
        "original_source_preserved": True,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print("Baseline file exists: YES")
    print("Expected candidate file exists: NO")
    print("Original dataset preserved: YES")
    print(f"\nExpected missing file: {injected_file}")
    print(f"Manifest: {manifest_file}")
    return manifest


def inject_f6_late(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
    *,
    seed: int = 6170,
    experiment_id: str | None = None,
    delay_minutes: int = 30,
) -> dict:
    """Simulate a candidate arriving a configurable number of minutes late."""
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    _validate_seed(seed)
    _validate_experiment_id(experiment_id)
    if type(delay_minutes) is not int or not 0 < delay_minutes <= 10080:
        raise ValueError("Delay must be an integer from 1 to 10080 minutes.")

    data_dir = Path(data_dir)
    source_file = (
        data_dir / "raw" / "yellow_taxi" / f"yellow_tripdata_{month}.parquet"
    )
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    experiment_dir = data_dir / "experiments" / month / "F6_late_file"
    if experiment_id is not None:
        experiment_dir = experiment_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    event_file = experiment_dir / "arrival_events.json"
    manifest_file = experiment_dir / "manifest.json"
    if any(
        p.exists()
        for p in (baseline_file, injected_file, event_file, manifest_file)
    ):
        raise FileExistsError(f"Experiment already exists: {experiment_dir}")

    print("\nF6B — LATE FILE ARRIVAL INJECTION")
    print("=" * 45)
    con = duckdb.connect(database=":memory:")
    try:
        print("Creating baseline sample...")
        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                USING SAMPLE {sample_size} ROWS (reservoir, {seed})
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        con.close()

    # The data is identical; F6B changes only the operational arrival times.
    shutil.copyfile(baseline_file, injected_file)
    baseline_rows = pq.read_metadata(baseline_file).num_rows
    injected_rows = pq.read_metadata(injected_file).num_rows
    if baseline_rows == 0 or baseline_rows != injected_rows:
        raise ValueError("F6B row-count verification failed.")
    if baseline_file.read_bytes() != injected_file.read_bytes():
        raise ValueError("F6B unexpectedly modified file contents.")

    observed_at = datetime.now(timezone.utc)
    # Keep candidate arrival before the observation even for long delays.
    expected_at = observed_at - timedelta(minutes=delay_minutes + 15)
    baseline_arrived_at = expected_at - timedelta(minutes=10)
    candidate_arrived_at = expected_at + timedelta(minutes=delay_minutes)
    actual_delay = (candidate_arrived_at - expected_at).total_seconds() / 60

    # These are operational observations, not ground-truth failure labels.
    arrival_events = {
        "expected_by_utc": expected_at.isoformat(),
        "reference_arrived_at_utc": baseline_arrived_at.isoformat(),
        "candidate_arrived_at_utc": candidate_arrived_at.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
    }
    event_file.write_text(json.dumps(arrival_events, indent=2), encoding="utf-8")

    manifest = {
        "failure_id": "F6",
        "failure_type": "missing_or_late_file",
        "scenario": "late_file",
        "month": month,
        "experiment_id": experiment_id,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "arrival_events_file": str(event_file),
        "sample_size": baseline_rows,
        "injected_at_utc": observed_at.isoformat(),
        "injection": {
            "operation": "simulate_late_arrival",
            "expected_delay_minutes": delay_minutes,
            "observed_delay_minutes": actual_delay,
            "random_seed": seed,
        },
        "file_exists": True,
        "schema_preserved": True,
        "row_count_preserved": True,
        "contents_preserved": True,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print("Candidate file exists: YES")
    print("Contents preserved: YES")
    print(f"Simulated arrival delay: {actual_delay:.0f} minutes")
    print(f"\nArrival events: {event_file}")
    print(f"Manifest: {manifest_file}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject controlled data pipeline failures."
    )
    parser.add_argument(
        "--failure", choices=["F1", "F2", "F3", "F4", "F5", "F6"],
        default="F1", help="Failure type to inject"
    )
    parser.add_argument("--f6-scenario", choices=["missing", "late"], default="missing")
    parser.add_argument("--month", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=6170)
    parser.add_argument("--experiment-id", default=None)

    # F1 configurable rename.
    parser.add_argument("--original-column", default="trip_distance")
    parser.add_argument("--new-column", default="distance")

    # F2 configurable type change.
    parser.add_argument("--column", default="total_amount")
    parser.add_argument("--target-type", default="VARCHAR")

    # F3/F5 use separate default injection rates.
    parser.add_argument("--injection-fraction", type=float, default=None)
    parser.add_argument("--mode", choices=["single", "correlated", "vendor"], default="single")
    parser.add_argument("--vendor-id", type=int, default=None)

    # F4 row loss and F6 late arrival.
    parser.add_argument("--removal-fraction", type=float, default=0.40)
    parser.add_argument("--delay-minutes", type=int, default=30)
    args = parser.parse_args()

    common = dict(
        month=args.month,
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        experiment_id=args.experiment_id,
    )
    if args.failure == "F1":
        inject_f1(
            **common,
            original_column=args.original_column,
            new_column=args.new_column,
        )
    elif args.failure == "F2":
        inject_f2(
            **common,
            column=args.column,
            target_type=args.target_type,
        )
    elif args.failure == "F3":
        inject_f3(
            **common,
            injection_fraction=(
                0.30 if args.injection_fraction is None else args.injection_fraction
            ),
            mode=args.mode,
            vendor_id=args.vendor_id,
        )
    elif args.failure == "F4":
        inject_f4(**common, removal_fraction=args.removal_fraction)
    elif args.failure == "F5":
        inject_f5(
            **common,
            injection_fraction=(
                0.05 if args.injection_fraction is None else args.injection_fraction
            ),
        )
    elif args.f6_scenario == "missing":
        inject_f6(**common)
    else:
        inject_f6_late(**common, delay_minutes=args.delay_minutes)


if __name__ == "__main__":
    main()
