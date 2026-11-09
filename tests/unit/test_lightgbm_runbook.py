"""Safeguards for the LightGBM diagnosis + runbook recommendation bridge."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from pipefail.contracts.schemas import FailureType, RepairAction
from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook


FEATURE_NAMES = [
    "file_missing", "schema_comparison_available", "missing_column_count",
    "added_column_count", "type_change_count", "row_count_available",
    "row_count_change_pct", "max_null_rate_increase_pp",
    "max_null_rate_decrease_pp", "invalid_rate_available",
    "invalid_distance_increase_pp", "distance_probe_failed",
    "revenue_probe_failed", "arrival_available", "late_arrival_minutes", "late_arrival_flag",
]


@pytest.fixture
def demo_model(tmp_path):
    # Small actual LightGBM model, not a mock: verify on-disk model load contract.
    rng = np.random.default_rng(6170)
    values = rng.normal(size=(90, len(FEATURE_NAMES)))
    X = pd.DataFrame(values, columns=FEATURE_NAMES)
    y = ["CLEAN", "F1", "F2", "F3", "F4", "F5", "F6"] * 12 + ["CLEAN"] * 6
    model = lgb.LGBMClassifier(n_estimators=7, n_jobs=1, verbosity=-1,
                               min_child_samples=2, random_state=6170)
    model.fit(X, y)
    path = tmp_path / "model.txt"
    manifest = tmp_path / "run_manifest.json"
    model.booster_.save_model(str(path))
    manifest.write_text(json.dumps({"features": FEATURE_NAMES,
                                    "label_order": ["CLEAN", "F1", "F2", "F3", "F4", "F5", "F6"]}),
                        encoding="utf-8")
    return path, manifest, dict(X.iloc[0])


def test_uses_saved_model_and_runbook(demo_model):
    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest)
    diagnosis, proposal = recommender.recommend(features, "test-01")
    assert diagnosis.model_name == "LIGHTGBM_RUNBOOK"
    assert diagnosis.confidence is not None
    assert 0 <= diagnosis.confidence <= 1
    assert diagnosis.failure_type in set(FailureType)
    assert proposal.status in {"NO_ACTION", "REVIEW_REQUIRED"}
    assert proposal.status != "SANDBOX_ELIGIBLE"  # no full raw evidence
    assert proposal.human_approval_required


def test_no_fabricated_type_conversion(demo_model):
    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest)
    # Make the model output F2 deterministically to verify reuse of repair guards.
    recommender.model.predict = lambda X: np.array([[0.01, 0.01, 0.94, 0.01, 0.01, 0.01, 0.01]])
    diagnosis, proposal = recommender.recommend(features, "test-F2")
    assert diagnosis.failure_type is FailureType.F2
    assert diagnosis.recommended_action is RepairAction.MANUAL_REVIEW
    assert proposal.status == "REVIEW_REQUIRED"
    assert proposal.action is RepairAction.MANUAL_REVIEW


def test_verified_f2_evidence_only_enables_sandbox_trial(demo_model):
    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest)
    recommender.model.predict = lambda X: np.array([[0.01, 0.01, 0.94, 0.01, 0.01, 0.01, 0.01]])
    evidence = {
        "reference": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "double"}},
        "candidate": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "string"}},
        "schema_diff": {"missing_columns": [], "added_columns": [],
                        "type_changes": [{"column": "total_amount"}]},
    }
    diagnosis, proposal = recommender.recommend(features, "test-F2", raw_evidence=evidence)
    assert diagnosis.failure_type is FailureType.F2
    assert proposal.status == "SANDBOX_ELIGIBLE"
    assert proposal.action == RepairAction.CONVERT_TYPE
    assert proposal.human_approval_required
    assert proposal.parameters["target_type"] == "double"


def test_low_score_defers_instead_of_repair(demo_model):
    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest, min_confidence=0.9)
    recommender.model.predict = lambda X: np.array([[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]])
    diagnosis, proposal = recommender.recommend(features, "low-score")
    assert diagnosis.failure_type is FailureType.UNKNOWN
    assert proposal.status == "REVIEW_REQUIRED"
    assert proposal.action is RepairAction.MANUAL_REVIEW


def test_missing_or_bad_feature_refused(demo_model):
    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest)
    with pytest.raises(ValueError, match="Missing model feature"):
        recommender.recommend({k: v for k, v in features.items() if k != "file_missing"}, "missing")
    with pytest.raises(ValueError, match="Non-finite"):
        recommender.recommend({**features, "file_missing": float("nan")}, "nan")


def test_mismatching_feature_order_is_refused(demo_model):
    path, manifest, _ = demo_model
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["features"] = list(reversed(payload["features"]))
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="feature names/order"):
        LightGBMRunbook(path, manifest)


def test_full_lightgbm_f2_sandbox_and_review_gate(demo_model, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    pq = pytest.importorskip("pyarrow.parquet")
    from pipefail.workflows.lightgbm_workflow import run_lightgbm_workflow

    path, manifest, features = demo_model
    recommender = LightGBMRunbook(path, manifest)
    recommender.model.predict = lambda X: np.array([[0.01, 0.01, 0.94, 0.01, 0.01, 0.01, 0.01]])
    clean = pa.table({"trip_distance": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
                      "total_amount": pa.array([5.5, 6.5, 7.5], type=pa.float64())})
    changed = clean.set_column(
        clean.schema.get_field_index("total_amount"), "total_amount",
        pc.cast(clean.column("total_amount"), pa.string(), safe=True),
    )
    ref_path = tmp_path / "reference.parquet"
    candidate_path = tmp_path / "candidate.parquet"
    pq.write_table(clean, ref_path)
    pq.write_table(changed, candidate_path)
    original = candidate_path.read_bytes()
    evidence = {
        "reference": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "double"}},
        "candidate": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "string"}},
        "schema_diff": {"missing_columns": [], "added_columns": [],
                        "type_changes": [{"column": "total_amount"}]},
    }
    record = run_lightgbm_workflow(
        recommender=recommender, incident_id="f2-lightgbm-demo", features=features,
        output_dir=tmp_path / "audit", raw_evidence=evidence,
        candidate_parquet=candidate_path, clean_reference_parquet=ref_path,
    )
    assert record["diagnosis"]["failure_type"] == "F2"
    assert record["proposal"]["action"] == "CONVERT_TYPE"
    assert record["sandbox"]["status"] == "SUCCEEDED"
    assert record["validation"]["status"] == "PASSED"
    assert record["review_status"] == "PENDING_HUMAN_REVIEW"
    assert record["live_data_modified"] is False
    assert candidate_path.read_bytes() == original
    with pytest.raises(FileExistsError):
        run_lightgbm_workflow(recommender=recommender, incident_id="f2-lightgbm-demo",
                              features=features, output_dir=tmp_path / "audit")
