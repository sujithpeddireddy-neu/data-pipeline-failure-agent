import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipefail.injection.failure_injector import inject_f3


@pytest.mark.parametrize(
    "mode,month,ratecode_type,vendor_id",
    [
        ("single", "2019-01", pa.float64(), None),
        ("correlated", "2025-01", pa.int64(), None),
        ("vendor", "2025-01", pa.int64(), 2),
    ],
)
def test_randomized_f3_injection(
    tmp_path,
    mode,
    month,
    ratecode_type,
    vendor_id,
):
    # Create an isolated synthetic TLC dataset.
    source_dir = tmp_path / "raw" / "yellow_taxi"
    source_dir.mkdir(parents=True)

    source_file = (
        source_dir / f"yellow_tripdata_{month}.parquet"
    )

    row_count = 2000

    table = pa.table({
        "VendorID": pa.array(
            [1 if i % 2 == 0 else 2 for i in range(row_count)],
            type=pa.int64(),
        ),
        "RatecodeID": pa.array(
            [None if i % 10 == 0 else 1 for i in range(row_count)],
            type=ratecode_type,
        ),
        "passenger_count": pa.array(
            [None if i % 10 == 0 else 2 for i in range(row_count)],
            type=pa.int64(),
        ),
        "trip_distance": pa.array(
            [float(i % 50) for i in range(row_count)],
            type=pa.float64(),
        ),
    })

    pq.write_table(table, source_file)

    original_hash = hashlib.sha256(
        source_file.read_bytes()
    ).hexdigest()

    # Generate the randomized F3 incident.
    manifest = inject_f3(
        month=month,
        data_dir=tmp_path,
        sample_size=1000,
        injection_fraction=0.20,
        seed=123,
        experiment_id=f"test-{mode}",
        mode=mode,
        vendor_id=vendor_id,
    )

    experiment_dir = (
        tmp_path
        / "experiments"
        / month
        / "F3_null_spike"
        / f"test-{mode}"
    )

    baseline_file = experiment_dir / "baseline.parquet"
    injected_file = experiment_dir / "injected.parquet"

    baseline = pq.read_table(baseline_file)
    injected = pq.read_table(injected_file)

    # Verify row count and schema preservation.
    assert baseline.num_rows == 1000
    assert injected.num_rows == 1000
    assert baseline.schema == injected.schema

    # Find records where new nulls were introduced.
    old_ratecodes = baseline["RatecodeID"].to_pylist()
    new_ratecodes = injected["RatecodeID"].to_pylist()

    ratecode_changes = {
        i
        for i, (old, new) in enumerate(
            zip(old_ratecodes, new_ratecodes)
        )
        if old is not None and new is None
    }

    old_passengers = baseline["passenger_count"].to_pylist()
    new_passengers = injected["passenger_count"].to_pylist()

    passenger_changes = {
        i
        for i, (old, new) in enumerate(
            zip(old_passengers, new_passengers)
        )
        if old is not None and new is None
    }

    # Verify the expected number of modified records.
    injection = manifest["injection"]

    expected_count = max(
        1,
        int(injection["eligible_record_count"] * 0.20),
    )

    assert len(ratecode_changes) == expected_count
    assert injection["modified_records"] == expected_count

    # Verify mode-specific behavior.
    if mode == "single":
        assert len(passenger_changes) == 0

    else:
        # Both columns must change on exactly the same rows.
        assert ratecode_changes == passenger_changes

    if mode == "vendor":
        vendors = baseline["VendorID"].to_pylist()

        # Every modified record must belong to VendorID 2.
        assert all(
            vendors[i] == 2
            for i in ratecode_changes
        )

    # Verify other columns remain unchanged.
    assert baseline["VendorID"].equals(injected["VendorID"])
    assert baseline["trip_distance"].equals(
        injected["trip_distance"]
    )

    # Verify the original source was never modified.
    assert hashlib.sha256(
        source_file.read_bytes()
    ).hexdigest() == original_hash