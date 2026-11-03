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


def inject_f1(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
) -> dict:
    """Inject a column-renaming failure into a sampled dataset."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

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

    if "trip_distance" not in source_schema.names:
        raise ValueError(
            "Source already lacks the expected trip_distance column."
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
                LIMIT {sample_size}
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
                RENAME (trip_distance AS distance)
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

    if (
        "trip_distance" not in original_columns
        or "trip_distance" in injected_columns
        or "distance" not in injected_columns
    ):
        raise ValueError("F1 schema injection verification failed.")

    expected_columns = [
        "distance" if column == "trip_distance" else column
        for column in original_columns
    ]

    if injected_columns != expected_columns:
        raise ValueError("Unexpected additional schema changes detected.")

    # Verify the rename preserved the column's data type.
    if (
        baseline_schema.field("trip_distance").type
        != injected_schema.field("distance").type
    ):
        raise ValueError("F1 unexpectedly changed the column data type.")

    # Preserve information required for later evaluation.
    manifest = {
        "failure_id": "F1",
        "failure_type": "schema_change",
        "month": month,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "rename_column",
            "original_column": "trip_distance",
            "new_column": "distance",
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
    print("Original column: trip_distance")
    print("Modified column: distance")
    print("Row count preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest

def inject_f2(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
) -> dict:
    """Inject an unexpected data type change."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

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
        / "F2_type_change"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("\nF2 — DATA TYPE CHANGE INJECTION")
    print("=" * 45)
    print(f"Source month: {month}")
    print(f"Sample size: {sample_size:,}")

    con = duckdb.connect(database=":memory:")

    try:
        print("\nCreating baseline sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                LIMIT {sample_size}
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        print("Injecting data type change...")

        con.execute(
            f"""
            COPY (
                SELECT
                    * REPLACE (
                        CAST(total_amount AS VARCHAR)
                        AS total_amount
                    )
                FROM read_parquet('{sql_path(baseline_file)}')
            )
            TO '{sql_path(injected_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

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

    original_type = str(
        baseline_schema.field("total_amount").type
    )

    modified_type = str(
        injected_schema.field("total_amount").type
    )

    if original_type != "double" or modified_type != "string":
        raise ValueError(
            f"Unexpected F2 types: {original_type} -> {modified_type}"
        )

    for column in baseline_schema.names:
        if column == "total_amount":
            continue

        if (
            baseline_schema.field(column).type
            != injected_schema.field(column).type
        ):
            raise ValueError(
                f"F2 unexpectedly changed type of {column}"
            )

    manifest = {
        "failure_id": "F2",
        "failure_type": "data_type_change",
        "month": month,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "change_column_type",
            "column": "total_amount",
            "original_type": original_type,
            "modified_type": modified_type,
        },
        "expected_failure": "incompatible_numeric_aggregation",
        "row_count_preserved": True,
        "column_names_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print("Modified column: total_amount")
    print(f"Type change: {original_type} -> {modified_type}")
    print("Row count preserved: YES")
    print(f"\nManifest: {manifest_file}")

    return manifest
def inject_f3(
    month: str,
    data_dir: Path,
    sample_size: int = 10000,
) -> dict:
    """Inject a 30% null-rate spike into RatecodeID."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

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
                LIMIT {sample_size}
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        baseline_rows = pq.read_metadata(baseline_file).num_rows

        # Deterministically select the first 30% of sampled rows.
        target_count = max(1, int(baseline_rows * 0.30))

        print("Injecting null-rate spike...")

        con.execute(
            f"""
            COPY (
                SELECT
                    * REPLACE (
                        CASE
                            WHEN ROW_NUMBER() OVER () <= {target_count}
                            THEN CAST(NULL AS DOUBLE)
                            ELSE RatecodeID
                        END AS RatecodeID
                    )
                FROM read_parquet('{sql_path(baseline_file)}')
            )
            TO '{sql_path(injected_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        baseline_nulls = con.execute(
            "SELECT COUNT_IF(RatecodeID IS NULL) "
            "FROM read_parquet(?)",
            [str(baseline_file)],
        ).fetchone()[0]

        injected_nulls = con.execute(
            "SELECT COUNT_IF(RatecodeID IS NULL) "
            "FROM read_parquet(?)",
            [str(injected_file)],
        ).fetchone()[0]

    finally:
        con.close()

    # ----------------------------------------
    # Restore original Parquet schema
    # ----------------------------------------

    # DuckDB may infer NOT NULL constraints when
    # writing a transformed Parquet dataset.
    # F3 must preserve the reference schema exactly.

    baseline_schema = pq.read_schema(baseline_file)

    injected_table = pq.read_table(injected_file)

    # Ensure column names and physical data types
    # haven't changed during injection.
    if injected_table.schema.names != baseline_schema.names:
        raise ValueError(
            "F3 unexpectedly changed column names."
        )

    for original, modified in zip(
        baseline_schema,
        injected_table.schema,
    ):
        if original.type != modified.type:
            raise ValueError(
                f"F3 unexpectedly changed type of {original.name}"
            )

    # Restore original field nullability and metadata.
    normalized_table = injected_table.cast(
        baseline_schema,
        safe=True,
    )

    normalized_file = (
        experiment_dir / "injected_normalized.parquet"
    )

    try:
        pq.write_table(
            normalized_table,
            normalized_file,
            compression="zstd",
        )

        normalized_file.replace(injected_file)

    finally:
        normalized_file.unlink(missing_ok=True)

    injected_rows = pq.read_metadata(injected_file).num_rows

    if baseline_rows != injected_rows:
        raise ValueError("F3 unexpectedly changed row count.")

    if pq.read_schema(baseline_file) != pq.read_schema(injected_file):
        raise ValueError("F3 unexpectedly changed the schema.")

    baseline_rate = baseline_nulls / baseline_rows
    injected_rate = injected_nulls / injected_rows

    if injected_rate < 0.30 or injected_nulls <= baseline_nulls:
        raise ValueError("F3 null-rate injection verification failed.")

    manifest = {
        "failure_id": "F3",
        "failure_type": "null_rate_spike",
        "month": month,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "sample_size": baseline_rows,
        "injected_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "injection": {
            "operation": "introduce_null_values",
            "column": "RatecodeID",
            "target_null_rate": 0.30,
            "baseline_null_rate": baseline_rate,
            "observed_null_rate": injected_rate,
        },
        "baseline_null_count": baseline_nulls,
        "injected_null_count": injected_nulls,
        "row_count_preserved": True,
        "schema_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print(f"Modified column: RatecodeID")
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
) -> dict:
    """Inject a 40% row-count reduction."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

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
                LIMIT {sample_size}
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

    # Remove 40% of records deterministically.
    removal_fraction = 0.40

    removed_rows = int(baseline_rows * removal_fraction)
    retained_rows = baseline_rows - removed_rows

    # Deterministic selection ensures reproducibility.
    rng = random.Random(6170)

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
            "random_seed": 6170,
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
) -> dict:
    """Inject negative trip distances into 5% of sampled records."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size < 10:
        raise ValueError("Sample size must be at least 10.")

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
                LIMIT {sample_size}
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

    injection_fraction = 0.05
    target_count = max(1, int(baseline_rows * injection_fraction))

    if len(eligible_indices) < target_count:
        raise ValueError(
            "Not enough valid records for F5 injection."
        )

    # Step 3: Select records reproducibly.
    rng = random.Random(6170)

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
            "target_fraction": injection_fraction,
            "modified_records": changed_count,
            "random_seed": 6170,
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
) -> dict:
    """
    Simulate a missing expected Parquet file.

    Creates a valid reference sample and records an
    expected candidate path that deliberately has no file.

    Original TLC data is never modified.
    """

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

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
        / "F6_missing_file"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    manifest_file = experiment_dir / "manifest.json"

    # Protect existing experiments.
    if any(
        path.exists()
        for path in (
            baseline_file,
            injected_file,
            manifest_file,
        )
    ):
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

    print("\nF6 — MISSING FILE INJECTION")
    print("=" * 45)

    # Create a valid reference dataset.
    con = duckdb.connect(database=":memory:")

    try:
        print("Creating baseline sample...")

        con.execute(
            f"""
            COPY (
                SELECT *
                FROM read_parquet('{sql_path(source_file)}')
                LIMIT {sample_size}
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

    # F6 simulation:
    # The expected candidate path deliberately remains absent.
    print("Simulating missing expected input file...")

    if injected_file.exists():
        raise ValueError(
            "F6 failed: candidate file unexpectedly exists."
        )

    check_time = datetime.now(timezone.utc).isoformat()

    # Record ground truth separately from evidence.
    manifest = {
        "failure_id": "F6",
        "failure_type": "missing_or_late_file",
        "scenario": "missing_file",
        "month": month,
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
        },
        "expected_failure": "file_not_found",
        "baseline_exists": True,
        "candidate_exists": False,
        "original_source_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

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
) -> dict:
    """Simulate a file arriving 30 minutes after its deadline."""

    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", month):
        raise ValueError("Month must use YYYY-MM format.")

    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")

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
        / "F6_late_file"
    )

    experiment_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"
    event_file = experiment_dir / "arrival_events.json"
    manifest_file = experiment_dir / "manifest.json"

    if any(
        p.exists()
        for p in (
            baseline_file,
            injected_file,
            event_file,
            manifest_file,
        )
    ):
        raise FileExistsError(
            f"Experiment already exists: {experiment_dir}"
        )

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
                LIMIT {sample_size}
            )
            TO '{sql_path(baseline_file)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    finally:
        con.close()

    # An identical copy represents the late-arriving file.
    # No data corruption or schema change is introduced.
    shutil.copyfile(baseline_file, injected_file)

    baseline_rows = pq.read_metadata(baseline_file).num_rows
    injected_rows = pq.read_metadata(injected_file).num_rows

    if baseline_rows == 0 or baseline_rows != injected_rows:
        raise ValueError("F6B row-count verification failed.")

    if baseline_file.read_bytes() != injected_file.read_bytes():
        raise ValueError("F6B unexpectedly modified file contents.")

    # Simulate arrival timing.
    observed_at = datetime.now(timezone.utc)

    expected_at = observed_at - timedelta(minutes=45)

    baseline_arrived_at = expected_at - timedelta(minutes=10)

    candidate_arrived_at = expected_at + timedelta(minutes=30)

    delay_minutes = (
        candidate_arrived_at - expected_at
    ).total_seconds() / 60

    # Operational timing evidence.
    # Do not include the ground-truth failure label here.
    arrival_events = {
        "expected_by_utc": expected_at.isoformat(),
        "reference_arrived_at_utc": baseline_arrived_at.isoformat(),
        "candidate_arrived_at_utc": candidate_arrived_at.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
    }

    event_file.write_text(
        json.dumps(arrival_events, indent=2),
        encoding="utf-8",
    )

    # Ground truth is stored separately.
    manifest = {
        "failure_id": "F6",
        "failure_type": "missing_or_late_file",
        "scenario": "late_file",
        "month": month,
        "source_file": str(source_file),
        "baseline_file": str(baseline_file),
        "injected_file": str(injected_file),
        "arrival_events_file": str(event_file),
        "sample_size": baseline_rows,
        "injected_at_utc": observed_at.isoformat(),
        "injection": {
            "operation": "simulate_late_arrival",
            "expected_delay_minutes": 30,
            "observed_delay_minutes": delay_minutes,
        },
        "file_exists": True,
        "schema_preserved": True,
        "row_count_preserved": True,
        "contents_preserved": True,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\nINJECTION SUCCESSFUL")
    print("=" * 45)
    print(f"Baseline records: {baseline_rows:,}")
    print(f"Injected records: {injected_rows:,}")
    print("Candidate file exists: YES")
    print("Contents preserved: YES")
    print(f"Simulated arrival delay: {delay_minutes:.0f} minutes")
    print(f"\nArrival events: {event_file}")
    print(f"Manifest: {manifest_file}")

    return manifest
def main():
    parser = argparse.ArgumentParser(
        description="Inject controlled data pipeline failures."
    )
    parser.add_argument(
        "--failure",
        choices=["F1", "F2", "F3", "F4", "F5", "F6"],
        default="F1",
        help="Failure type to inject",
    )
    parser.add_argument(
        "--f6-scenario",
        choices=["missing", "late"],
        default="missing",
        help="Select the F6 availability scenario",
    )
    parser.add_argument("--month", required=True)

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    if args.failure == "F1":
        inject_f1(
            month=args.month,
            data_dir=args.data_dir,
            sample_size=args.sample_size,
        )

    elif args.failure == "F2":
        inject_f2(
            month=args.month,
            data_dir=args.data_dir,
            sample_size=args.sample_size,
        )

    elif args.failure == "F3":
        inject_f3(
            month=args.month,
            data_dir=args.data_dir,
            sample_size=args.sample_size,
        )

    elif args.failure == "F4":
        inject_f4(
            month=args.month,
            data_dir=args.data_dir,
            sample_size=args.sample_size,
        )

    elif args.failure == "F5":
        inject_f5(
            month=args.month,
            data_dir=args.data_dir,
            sample_size=args.sample_size,
        )

    elif args.failure == "F6":

        if args.f6_scenario == "missing":
            inject_f6(
                month=args.month,
                data_dir=args.data_dir,
                sample_size=args.sample_size,
            )

        elif args.f6_scenario == "late":
            inject_f6_late(
                month=args.month,
                data_dir=args.data_dir,
                sample_size=args.sample_size,
            )

if __name__ == "__main__":
    main()