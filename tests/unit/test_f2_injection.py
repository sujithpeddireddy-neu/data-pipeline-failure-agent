"""F2 configurable datatype-change regression tests."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f2


@pytest.fixture
def sample_data(tmp_path):
    source_dir = tmp_path / "raw" / "yellow_taxi"
    source_dir.mkdir(parents=True)
    source = source_dir / "yellow_tripdata_2025-01.parquet"
    row_count = 2000
    table = pa.table(
        {
            "row_id": pa.array(range(row_count), type=pa.int64()),
            "VendorID": pa.array([i % 2 + 1 for i in range(row_count)], type=pa.int64()),
            "trip_distance": pa.array([float(i % 50 + 1) for i in range(row_count)], type=pa.float64()),
            "total_amount": pa.array([float(i % 100 + 2) for i in range(row_count)], type=pa.float64()),
            "RatecodeID": pa.array([1] * row_count, type=pa.int64()),
            "passenger_count": pa.array([2] * row_count, type=pa.int64()),
        }
    )
    pq.write_table(table, source)
    return tmp_path, source, hashlib.sha256(source.read_bytes()).hexdigest()


def check_original_preserved(source, expected_hash):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash


@pytest.mark.parametrize(
    "column,target_type,expected_arrow_type",
    [
        ("total_amount", "VARCHAR", "string"),
        ("trip_distance", "BIGINT", "int64"),
    ],
)
def test_f2_configurable_type_change(sample_data, column, target_type, expected_arrow_type):
    data_dir, source, source_hash = sample_data
    manifest = inject_f2(
        "2025-01", data_dir, sample_size=1000, seed=456,
        experiment_id=f"type-{target_type.lower()}",
        column=column, target_type=target_type,
    )
    d = data_dir / "experiments" / "2025-01" / "F2_type_change" / f"type-{target_type.lower()}"
    before, after = pq.read_table(d / "baseline.parquet"), pq.read_table(d / "injected.parquet")
    assert before.num_rows == after.num_rows == 1000
    assert before.column_names == after.column_names
    assert str(before.schema.field(column).type) != expected_arrow_type
    assert str(after.schema.field(column).type) == expected_arrow_type
    for n in before.column_names:
        if n != column:
            assert before[n].equals(after[n])
    assert manifest["injection"]["random_seed"] == 456
    check_original_preserved(source, source_hash)
