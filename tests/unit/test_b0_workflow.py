"""B0 workflow tests. Samples are generated locally; never download DVC data."""

import json
from pathlib import Path

import pytest

from pipefail.contracts.schemas import FailureType, RepairAction
from pipefail.detection.rules import detect_failure
from pipefail.diagnosis.runbook import diagnose_failure
from pipefail.repair.actions import propose_repair
from pipefail.repair.sandbox import execute_in_sandbox
from pipefail.repair.validator import validate_sandbox
from pipefail.review.approval import record_review
from pipefail.workflows.orchestrator import run_b0_workflow


BASE = dict(
    file_missing=0, schema_comparison_available=1,
    missing_column_count=0, added_column_count=0, type_change_count=0,
    row_count_available=1, row_count_change_pct=0.0,
    max_null_rate_increase_pp=0.0, max_null_rate_decrease_pp=0.0,
    invalid_rate_available=1, invalid_distance_increase_pp=0.0,
    distance_probe_failed=0, revenue_probe_failed=0,
    arrival_available=0, late_arrival_minutes=0.0, late_arrival_flag=0,
)


def schema_evidence():
    return {
        "reference": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "double"}},
        "candidate": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "string"}},
        "schema_diff": {"missing_columns": [], "added_columns": [],
                        "type_changes": [{"column": "total_amount"}]},
    }


def f2_proposal():
    diagnosis = diagnose_failure({**BASE, "type_change_count": 1}, "f2-demo")
    return propose_repair(diagnosis, schema_evidence())


def test_f2_requires_raw_evidence_and_only_approves_sandbox_trial():
    diag = diagnose_failure({**BASE, "type_change_count": 1}, "f2-demo")
    assert propose_repair(diag).status == "REVIEW_REQUIRED"
    proposal = propose_repair(diag, schema_evidence())
    assert proposal.action == RepairAction.CONVERT_TYPE
    assert proposal.status == "SANDBOX_ELIGIBLE"
    assert proposal.parameters["target_type"] == "double"


def test_conflicting_schema_cannot_be_repaired_automatically():
    diagnosis = diagnose_failure({**BASE, "type_change_count": 1}, "f2-demo")
    evidence = schema_evidence()
    evidence["schema_diff"]["added_columns"].append("random")
    assert propose_repair(diagnosis, evidence).status == "REVIEW_REQUIRED"


def test_f3_never_quarantines_rows():
    diagnosis = diagnose_failure({**BASE, "max_null_rate_increase_pp": 30}, "f3")
    assert diagnosis.failure_type is FailureType.F3
    proposal = propose_repair(diagnosis)
    assert proposal.status == "REVIEW_REQUIRED"
    assert proposal.action == RepairAction.MANUAL_REVIEW


def test_clean_produces_no_action():
    diagnosis = diagnose_failure(BASE, "clean")
    assert propose_repair(diagnosis).status == "NO_ACTION"


def test_unvalidated_cannot_be_approved(tmp_path):
    with pytest.raises(ValueError, match="without matching successful validation"):
        record_review(f2_proposal(), None, reviewer="Alex", decision="APPROVE",
                      notes="not tested", log_dir=tmp_path)


def test_review_requires_identity(tmp_path):
    with pytest.raises(ValueError, match="Reviewer"):
        record_review(f2_proposal(), None, reviewer="", decision="DEFER",
                      notes="need more evidence", log_dir=tmp_path)


def test_review_does_not_modify_live_data(tmp_path):
    result = record_review(f2_proposal(), None, reviewer="Alex", decision="DEFER",
                           notes="need independent reference", log_dir=tmp_path)
    assert result.deployment_performed is False
    assert (tmp_path / "f2-demo_review.json").is_file()
    with pytest.raises(FileExistsError):
        record_review(f2_proposal(), None, reviewer="Alex", decision="DEFER",
                      notes="duplicate", log_dir=tmp_path)


def test_unsupported_action_skips_sandbox(tmp_path):
    diagnosis = diagnose_failure({**BASE, "max_null_rate_increase_pp": 28}, "f3")
    result = execute_in_sandbox(propose_repair(diagnosis), tmp_path / "in.parquet", tmp_path)
    assert result.status == "SKIPPED"


def test_workflow_includes_audit_and_human_gate(tmp_path):
    run = run_b0_workflow(incident_id="f3", features={**BASE, "max_null_rate_increase_pp": 30},
                          output_dir=tmp_path)
    assert run["review_status"] == "PENDING_HUMAN_REVIEW"
    assert run["sandbox"] is None
    assert run["live_data_modified"] is False
    assert (tmp_path / "f3_audit.json").is_file()
    with pytest.raises(FileExistsError):
        run_b0_workflow(incident_id="f3", features=BASE, output_dir=tmp_path)


# These tests need pyarrow; it is already installed in the user's project venv.
@pytest.fixture
def parquet_pair(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    pq = pytest.importorskip("pyarrow.parquet")
    clean = pa.table({"trip_distance": pa.array([2.0, 3.0, 4.0], type=pa.float64()),
                      "total_amount": pa.array([5.25, 6.5, 7.75], type=pa.float64())})
    idx = clean.schema.get_field_index("total_amount")
    corrupted = clean.set_column(idx, "total_amount",
                                 pc.cast(clean.column(idx), pa.string(), safe=True))
    ref = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    pq.write_table(clean, ref)
    pq.write_table(corrupted, candidate)
    return ref, candidate


def test_f2_end_to_end_reversible_sandbox(parquet_pair, tmp_path):
    ref, candidate = parquet_pair
    original_bytes = candidate.read_bytes()
    record = run_b0_workflow(
        incident_id="f2-demo", features={**BASE, "type_change_count": 1},
        raw_evidence=schema_evidence(), candidate_parquet=candidate,
        clean_reference_parquet=ref, output_dir=tmp_path / "output",
    )
    assert record["diagnosis"]["failure_type"] == "F2"
    assert record["sandbox"]["status"] == "SUCCEEDED"
    assert record["validation"]["status"] == "PASSED"
    assert record["review_status"] == "PENDING_HUMAN_REVIEW"
    assert candidate.read_bytes() == original_bytes
    # Human must still act, and even approval does not deploy.
    from pipefail.repair.validator import ValidationResult
    validated = ValidationResult("f2-demo", "PASSED", True, ("matches",))
    reviewed = record_review(f2_proposal(), validated, reviewer="Alex",
                             decision="APPROVE", notes="demo only", log_dir=tmp_path / "reviews")
    assert reviewed.deployment_performed is False


def test_no_clean_reference_never_passes_validation(parquet_pair, tmp_path):
    _, candidate = parquet_pair
    trial = execute_in_sandbox(f2_proposal(), candidate, tmp_path / "trial")
    assert trial.status == "SUCCEEDED"
    result = validate_sandbox(trial, None)
    assert not result.passed
    with pytest.raises(ValueError):
        record_review(f2_proposal(), result, reviewer="Alex", decision="APPROVE",
                      notes="not safe", log_dir=tmp_path / "review")


def test_corrupted_other_column_detected(parquet_pair, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ref, candidate = parquet_pair
    altered = pq.read_table(ref).set_column(0, "trip_distance",
                                               pa.array([100.0, 3.0, 4.0]))
    wrong_ref = tmp_path / "wrong_ref.parquet"
    pq.write_table(altered, wrong_ref)
    trial = execute_in_sandbox(f2_proposal(), candidate, tmp_path / "trial")
    checked = validate_sandbox(trial, wrong_ref)
    assert checked.status == "FAILED"
    assert not checked.passed


def test_unparseable_type_change_fails_closed(parquet_pair, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    _, candidate = parquet_pair
    table = pq.read_table(candidate)
    idx = table.schema.get_field_index("total_amount")
    changed = table.set_column(idx, "total_amount",
                               pa.array(["5.25", "not-a-number", "7.75"]))
    pq.write_table(changed, candidate)
    trial = execute_in_sandbox(f2_proposal(), candidate, tmp_path / "trial")
    assert trial.status == "FAILED"
    assert not (tmp_path / "trial" / "f2-demo_trial.parquet").exists()
