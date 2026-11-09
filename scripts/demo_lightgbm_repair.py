"""Credential-free LightGBM -> runbook -> F2 sandbox validation demonstration.

Uses an F2 *training* feature vector to show integration, then generates a new
three-row synthetic Parquet pair. This is NOT a held-out repair-success test.
No original data is modified; approval remains pending.
"""

import argparse
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook
from pipefail.workflows.lightgbm_workflow import run_lightgbm_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_1200_v1"))
    parser.add_argument("--model-dir", type=Path, default=Path("results/walkthrough_v1/lightgbm"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/walkthrough_v1/lightgbm/f2_workflow_demo"))
    args = parser.parse_args()
    recommender = LightGBMRunbook(
        args.model_dir / "model.txt", args.model_dir / "run_manifest.json"
    )
    train = pd.read_csv(args.data_dir / "train.csv")
    f2 = train[train["label"] == "F2"]
    if f2.empty:
        raise ValueError("Training split has no F2 example")
    feature_row = dict(f2.iloc[0])
    # This is a sandbox-only generated example, not a real historical taxi file.
    clean = pa.table({
        "trip_distance": pa.array([1.0, 3.5, 2.25], type=pa.float64()),
        "total_amount": pa.array([12.50, 20.25, 16.75], type=pa.float64()),
    })
    idx = clean.schema.get_field_index("total_amount")
    changed = clean.set_column(idx, "total_amount",
                               pc.cast(clean.column(idx), pa.string(), safe=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=args.output_dir))
    clean_path = run_dir / "reference.parquet"
    candidate_path = run_dir / "candidate.parquet"
    pq.write_table(clean, clean_path)
    pq.write_table(changed, candidate_path)
    evidence = {
        "reference": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "double"}},
        "candidate": {"exists": True, "row_count": 3,
                      "schema": {"trip_distance": "double", "total_amount": "string"}},
        "schema_diff": {"missing_columns": [], "added_columns": [],
                        "type_changes": [{"column": "total_amount"}]},
    }
    result = run_lightgbm_workflow(
        recommender=recommender,
        incident_id="lightgbm-f2-sample",
        features=feature_row,
        output_dir=run_dir,
        raw_evidence=evidence,
        candidate_parquet=candidate_path,
        clean_reference_parquet=clean_path,
    )
    print("LightGBM diagnosis:", result["diagnosis"]["failure_type"])
    print("Runbook proposal:", result["proposal"]["action"],
          "|", result["proposal"]["status"])
    print("Sandbox:", result["sandbox"]["status"] if result["sandbox"] else "NOT_RUN")
    print("Validation:", result["validation"]["status"] if result["validation"] else "NOT_RUN")
    print("Review status:", result["review_status"])
    print("Live data modified:", result["live_data_modified"])
    print("Audit record:", run_dir / "lightgbm-f2-sample_audit.json")


if __name__ == "__main__":
    main()
