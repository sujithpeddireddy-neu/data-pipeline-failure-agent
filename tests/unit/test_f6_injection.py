"""F6 missing-file and late-arrival regression tests."""

import hashlib
import json
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f6, inject_f6_late


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


def test_f6_missing_file(sample_data):
    data_dir, source, source_hash = sample_data
    manifest = inject_f6(
        "2025-01", data_dir, sample_size=1000, seed=999,
        experiment_id="missing-001",
    )
    d = data_dir / "experiments" / "2025-01" / "F6_missing_file" / "missing-001"
    assert pq.read_metadata(d / "baseline.parquet").num_rows == 1000
    assert not (d / "injected.parquet").exists()
    assert manifest["scenario"] == "missing_file"
    assert manifest["injection"]["random_seed"] == 999
    check_original_preserved(source, source_hash)

def test_f6_configurable_late_arrival(sample_data):
    data_dir, source, source_hash = sample_data
    manifest = inject_f6_late(
        "2025-01", data_dir, sample_size=1000, seed=777,
        experiment_id="late-001", delay_minutes=90,
    )
    d = data_dir / "experiments" / "2025-01" / "F6_late_file" / "late-001"
    assert (d / "baseline.parquet").read_bytes() == (d / "injected.parquet").read_bytes()
    events = json.loads((d / "arrival_events.json").read_text(encoding="utf-8"))
    expected = datetime.fromisoformat(events["expected_by_utc"])
    arrived = datetime.fromisoformat(events["candidate_arrived_at_utc"])
    observed = datetime.fromisoformat(events["observed_at_utc"])
    assert (arrived - expected).total_seconds() == 90 * 60
    assert arrived <= observed
    assert manifest["injection"]["expected_delay_minutes"] == 90
    assert manifest["injection"]["random_seed"] == 777
    assert "failure_id" not in events
    check_original_preserved(source, source_hash)
