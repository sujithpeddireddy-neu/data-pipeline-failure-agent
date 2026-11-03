import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f4


def test_randomized_f4_injection(tmp_path):
    # Create an isolated source dataset.
    source_dir = tmp_path / "raw" / "yellow_taxi"
    source_dir.mkdir(parents=True)

    source = source_dir / "yellow_tripdata_2022-06.parquet"

    table = pa.table({
        "row_id": pa.array(range(2000), type=pa.int64()),
        "trip_distance": pa.array(
            [float(i % 50) for i in range(2000)],
            type=pa.float64(),
        ),
    })

    pq.write_table(table, source)

    original_hash = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    # Inject 15% record loss.
    manifest = inject_f4(
        month="2022-06",
        data_dir=tmp_path,
        sample_size=1000,
        removal_fraction=0.15,
        seed=123,
        experiment_id="test-001",
    )

    experiment_dir = (
        tmp_path
        / "experiments"
        / "2022-06"
        / "F4_row_count_anomaly"
        / "test-001"
    )

    baseline = experiment_dir / "baseline.parquet"
    injected = experiment_dir / "injected.parquet"

    # Verify record counts.
    assert pq.read_metadata(baseline).num_rows == 1000
    assert pq.read_metadata(injected).num_rows == 850

    # Verify schema preservation.
    assert pq.read_schema(baseline) == pq.read_schema(injected)

    # Verify manifest values.
    assert manifest["injection"]["target_removal_fraction"] == 0.15
    assert manifest["injection"]["observed_removal_fraction"] == 0.15
    assert manifest["injection"]["random_seed"] == 123

    # Verify that sampling does not simply take the first 1000 rows.
    sampled_ids = set(pq.read_table(baseline)["row_id"].to_pylist())
    assert sampled_ids != set(range(1000))

    # Verify that the source dataset was never modified.
    assert hashlib.sha256(
        source.read_bytes()
    ).hexdigest() == original_hash

    # Verify that existing incidents cannot be overwritten.
    with pytest.raises(FileExistsError):
        inject_f4(
            month="2022-06",
            data_dir=tmp_path,
            sample_size=1000,
            removal_fraction=0.15,
            seed=123,
            experiment_id="test-001",
        )