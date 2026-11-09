"""Regression tests for F2 nullability restoration and fail-closed behavior."""
import pytest

from pipefail.workflows.orchestrator import run_b0_workflow

BASE = {
    "file_missing": 0, "schema_comparison_available": 1,
    "missing_column_count": 0, "added_column_count": 0,
    "type_change_count": 1, "row_count_available": 1,
    "row_count_change_pct": 0.0, "max_null_rate_increase_pp": 0.0,
    "max_null_rate_decrease_pp": 0.0, "invalid_rate_available": 1,
    "invalid_distance_increase_pp": 0.0, "distance_probe_failed": 0,
    "revenue_probe_failed": 1, "arrival_available": 0,
    "late_arrival_minutes": 0.0, "late_arrival_flag": 0,
}


@pytest.fixture
def f2_nonnullable_pair(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    pq = pytest.importorskip("pyarrow.parquet")
    clean = pa.Table.from_arrays(
        [pa.array([1.5, 2.5], type=pa.float64())],
        schema=pa.schema([pa.field("trip_distance", pa.float64(), nullable=False)]),
    )
    changed = clean.set_column(
        0, pa.field("trip_distance", pa.string(), nullable=True),
        pc.cast(clean.column(0), pa.string(), safe=True),
    )
    ref, candidate = tmp_path / "reference.parquet", tmp_path / "candidate.parquet"
    pq.write_table(clean, ref)
    pq.write_table(changed, candidate)
    evidence = {
        "reference": {"exists": True, "row_count": 2,
                      "schema": {"trip_distance": "double"}},
        "candidate": {"exists": True, "row_count": 2,
                      "schema": {"trip_distance": "string"}},
        "schema_diff": {"missing_columns": [], "added_columns": [],
                        "type_changes": [{"column": "trip_distance"}]},
    }
    return ref, candidate, evidence


def test_missing_nullability_contract_fails_closed(f2_nonnullable_pair, tmp_path):
    ref, candidate, evidence = f2_nonnullable_pair
    record = run_b0_workflow(
        incident_id="without-contract", features=BASE, raw_evidence=evidence,
        candidate_parquet=candidate, clean_reference_parquet=ref,
        output_dir=tmp_path / "no-contract",
    )
    assert record["sandbox"]["status"] == "SUCCEEDED"
    assert record["validation"]["status"] == "FAILED"
    assert record["review_status"] == "BLOCKED_VALIDATION_FAILED"


def test_explicit_trusted_contract_restores_nonnullable_schema(f2_nonnullable_pair, tmp_path):
    ref, candidate, evidence = f2_nonnullable_pair
    original_bytes = candidate.read_bytes()
    record = run_b0_workflow(
        incident_id="with-contract", features=BASE, raw_evidence=evidence,
        candidate_parquet=candidate, clean_reference_parquet=ref,
        output_dir=tmp_path / "contract",
        trusted_nullability={"trip_distance": False},
        schema_contract_id="declared-before-incident-v1",
    )
    assert record["sandbox"]["status"] == "SUCCEEDED"
    assert record["validation"]["status"] == "PASSED"
    assert record["review_status"] == "PENDING_HUMAN_REVIEW"
    assert record["live_data_modified"] is False
    assert candidate.read_bytes() == original_bytes
    pq = pytest.importorskip("pyarrow.parquet")
    repaired = pq.read_table(record["sandbox"]["output_path"])
    assert repaired.schema.field("trip_distance").nullable is False


def test_nullability_override_requires_provenance(f2_nonnullable_pair, tmp_path):
    ref, candidate, evidence = f2_nonnullable_pair
    with pytest.raises(ValueError, match="schema_contract_id"):
        run_b0_workflow(
            incident_id="no-source", features=BASE, raw_evidence=evidence,
            candidate_parquet=candidate, clean_reference_parquet=ref,
            output_dir=tmp_path / "invalid",
            trusted_nullability={"trip_distance": False},
        )
