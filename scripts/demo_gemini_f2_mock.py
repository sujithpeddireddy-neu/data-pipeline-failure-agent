"""Credential-free B2 F2 sandbox integration demonstration."""

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook
from pipefail.workflows.gemini_workflow import run_gemini_workflow


class MockGemini:

    model = "MOCK_GEMINI_F2"
    last_attempts = 0
    last_latency_seconds = 0.0
    last_errors = []

    def diagnose(self, payload):

        return {
            "failure_type": "F2",
            "recommended_action": "CONVERT_TYPE",
            "affected_column": "total_amount",
            "reason": (
                "The total_amount column changed "
                "from double to string."
            ),
            "evidence": [
                "Reference type: double",
                "Candidate type: string",
                "Row count unchanged",
            ],
            "uncertainty": "low",
        }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():

    model_dir = Path(
        "results/smoke_runs/20261123T232846Z/lightgbm"
    )

    data_dir = Path("tests/fixtures/smoke_test")

    recommender = LightGBMRunbook(
        model_dir / "model.txt",
        model_dir / "run_manifest.json",
    )

    train = pd.read_csv(data_dir / "train.csv")

    f2 = train[train["label"] == "F2"]

    if f2.empty:
        raise ValueError("No F2 training incident found")

    row = f2.iloc[0].to_dict()

    features = {
        k: row[k]
        for k in recommender.feature_names
    }

    clean = pa.table({
        "trip_distance": pa.array(
            [1.0, 3.5, 2.25],
            type=pa.float64(),
        ),
        "total_amount": pa.array(
            [12.50, 20.25, 16.75],
            type=pa.float64(),
        ),
    })

    idx = clean.schema.get_field_index("total_amount")

    changed = clean.set_column(
        idx,
        "total_amount",
        pc.cast(
            clean.column(idx),
            pa.string(),
            safe=True,
        ),
    )

    output = Path("results/gemini_f2_mock")

    output.mkdir(parents=True, exist_ok=True)

    run_dir = Path(
        tempfile.mkdtemp(prefix="run_", dir=output)
    )

    clean_path = run_dir / "reference.parquet"
    candidate_path = run_dir / "candidate.parquet"

    pq.write_table(clean, clean_path)
    pq.write_table(changed, candidate_path)

    original_candidate_hash = file_hash(candidate_path)
    original_reference_hash = file_hash(clean_path)

    evidence = {
        "reference": {
            "exists": True,
            "row_count": 3,
            "schema": {
                "trip_distance": "double",
                "total_amount": "double",
            },
        },
        "candidate": {
            "exists": True,
            "row_count": 3,
            "schema": {
                "trip_distance": "double",
                "total_amount": "string",
            },
        },
        "schema_diff": {
            "missing_columns": [],
            "added_columns": [],
            "type_changes": [
                {"column": "total_amount"}
            ],
        },
    }

    result = run_gemini_workflow(
        recommender=recommender,
        gemini=MockGemini(),
        incident_id="b2-f2-mock",
        features=features,
        output_dir=run_dir,
        raw_evidence=evidence,
        candidate_parquet=candidate_path,
        clean_reference_parquet=clean_path,
    )

    summary = {
        "test_mode": "MOCK_GEMINI",
        "real_api_requests": 0,
        "lightgbm_diagnosis": result["lightgbm"]["label"],
        "gemini_diagnosis": result["diagnosis"]["failure_type"],
        "repair_action": result["proposal"]["action"],
        "proposal_status": result["proposal"]["status"],
        "sandbox_status": (
            result["sandbox"]["status"]
            if result["sandbox"] else "NOT_RUN"
        ),
        "validation_status": (
            result["validation"]["status"]
            if result["validation"] else "NOT_RUN"
        ),
        "review_status": result["review_status"],
        "live_data_modified": result["live_data_modified"],
        "candidate_unchanged": (
            file_hash(candidate_path)
            == original_candidate_hash
        ),
        "reference_unchanged": (
            file_hash(clean_path)
            == original_reference_hash
        ),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))

    assert summary["gemini_diagnosis"] == "F2"
    assert summary["repair_action"] == "CONVERT_TYPE"
    assert summary["proposal_status"] == "SANDBOX_ELIGIBLE"
    assert summary["sandbox_status"] == "SUCCEEDED"
    assert summary["validation_status"] == "PASSED"
    assert summary["review_status"] == "PENDING_HUMAN_REVIEW"
    assert summary["live_data_modified"] is False
    assert summary["candidate_unchanged"]
    assert summary["reference_unchanged"]

    print("\nPASS: B2 F2 sandbox integration verified.")
    print("Results saved to:", run_dir)


if __name__ == "__main__":
    main()
