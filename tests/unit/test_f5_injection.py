import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f5


def test_randomized_f5_injection(tmp_path):
    # Create an isolated synthetic dataset.
    source_dir = tmp_path / "raw" / "yellow_taxi"
    source_dir.mkdir(parents=True)

    source = source_dir / "yellow_tripdata_2025-01.parquet"

    # Include valid, negative, and null trip distances.
    distances = [
        None if i % 25 == 0
        else -2.0 if i % 20 == 0
        else float(i % 50 + 1)
        for i in range(2000)
    ]

    table = pa.table({
        "row_id": pa.array(range(2000), type=pa.int64()),
        "trip_distance": pa.array(
            distances, type=pa.float64()
        ),
        "total_amount": pa.array(
            [float(i % 100 + 2) for i in range(2000)],
            type=pa.float64(),
        ),
    })

    pq.write_table(table, source)

    original_hash = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    # Generate an F5 incident.
    manifest = inject_f5(
        month="2025-01",
        data_dir=tmp_path,
        sample_size=1000,
        injection_fraction=0.10,
        seed=123,
        experiment_id="test-001",
    )

    experiment_dir = (
        tmp_path
        / "experiments"
        / "2025-01"
        / "F5_invalid_values"
        / "test-001"
    )

    baseline = pq.read_table(
        experiment_dir / "baseline.parquet"
    )

    injected = pq.read_table(
        experiment_dir / "injected.parquet"
    )

    # Verify row counts and schema.
    assert baseline.num_rows == 1000
    assert injected.num_rows == 1000
    assert baseline.schema == injected.schema

    old_distances = baseline["trip_distance"].to_pylist()
    new_distances = injected["trip_distance"].to_pylist()

    eligible_indices = {
        i
        for i, value in enumerate(old_distances)
        if value is not None and value >= 0
    }

    changed_indices = {
        i
        for i, (old, new) in enumerate(
            zip(old_distances, new_distances)
        )
        if old != new
    }

    expected_count = max(
        1,
        int(len(eligible_indices) * 0.10)
    )

    # Verify exact injection severity.
    assert len(changed_indices) == expected_count

    # Only previously valid records should change.
    assert changed_indices.issubset(eligible_indices)

    # Every modified distance should become -1.0.
    assert all(
        new_distances[i] == -1.0
        for i in changed_indices
    )

    # Verify the actual increase in invalid records.
    baseline_invalid = sum(
        value is not None and value < 0
        for value in old_distances
    )

    injected_invalid = sum(
        value is not None and value < 0
        for value in new_distances
    )

    assert injected_invalid - baseline_invalid == expected_count

    # Verify unrelated columns remain unchanged.
    assert baseline["row_id"].equals(injected["row_id"])
    assert baseline["total_amount"].equals(
        injected["total_amount"]
    )

    # Verify randomized sampling.
    sampled_ids = set(
        baseline["row_id"].to_pylist()
    )
    assert sampled_ids != set(range(1000))

    # Verify manifest parameters.
    injection = manifest["injection"]

    assert injection["target_fraction"] == 0.10
    assert injection["random_seed"] == 123
    assert injection["eligible_record_count"] == len(
        eligible_indices
    )
    assert injection["modified_records"] == expected_count

    # Original dataset must remain unchanged.
    assert hashlib.sha256(
        source.read_bytes()
    ).hexdigest() == original_hash

    # Existing experiment files cannot be overwritten.
    with pytest.raises(FileExistsError):
        inject_f5(
            month="2025-01",
            data_dir=tmp_path,
            sample_size=1000,
            injection_fraction=0.10,
            seed=123,
            experiment_id="test-001",
        )