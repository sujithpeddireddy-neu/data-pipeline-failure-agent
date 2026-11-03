"""F1 configurable schema-rename regression test."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f1


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


def test_f1_configurable_rename(sample_data):
    data_dir, source, source_hash = sample_data
    manifest = inject_f1(
        "2025-01", data_dir, sample_size=1000,
        seed=123, experiment_id="rename-001",
        original_column="total_amount", new_column="total_fare",
    )
    d = data_dir / "experiments" / "2025-01" / "F1_schema_change" / "rename-001"
    before, after = pq.read_table(d / "baseline.parquet"), pq.read_table(d / "injected.parquet")
    assert before.num_rows == after.num_rows == 1000
    assert after.column_names == [
        "total_fare" if n == "total_amount" else n for n in before.column_names
    ]
    assert before["total_amount"].equals(after["total_fare"])
    assert manifest["injection"]["random_seed"] == 123
    assert set(before["row_id"].to_pylist()) != set(range(1000))
    check_original_preserved(source, source_hash)
    with pytest.raises(FileExistsError):
        inject_f1(
            "2025-01", data_dir, sample_size=1000, seed=123,
            experiment_id="rename-001", original_column="total_amount", new_column="total_fare"
        )
