"""Credential-free, small synthetic F2/F3 end-to-end B0 walkthrough.

Run: python scripts/demo_b0_workflow.py
Generates new isolated artifacts under results/b0_workflow_demo/run_*/.
Never changes source TLC files, approves a repair, or deploys a change.
"""

from pathlib import Path
import argparse
import json
import tempfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from pipefail.workflows.orchestrator import run_b0_workflow


CLEAN_FEATURES = {
    "file_missing": 0, "schema_comparison_available": 1,
    "missing_column_count": 0, "added_column_count": 0,
    "type_change_count": 0, "row_count_available": 1,
    "row_count_change_pct": 0., "max_null_rate_increase_pp": 0.,
    "max_null_rate_decrease_pp": 0., "invalid_rate_available": 1,
    "invalid_distance_increase_pp": 0., "distance_probe_failed": 0,
    "revenue_probe_failed": 0, "arrival_available": 0,
    "late_arrival_minutes": 0., "late_arrival_flag": 0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/b0_workflow_demo"))
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run_", dir=root))

    clean = pa.table({
        "trip_distance": pa.array([1.0, 3.5, 2.25], type=pa.float64()),
        "total_amount": pa.array([12.50, 20.25, 16.75], type=pa.float64()),
    })
    index = clean.schema.get_field_index("total_amount")
    altered = clean.set_column(index, "total_amount",
                               pc.cast(clean.column(index), pa.string(), safe=True))
    clean_path = run / "reference.parquet"
    damaged_path = run / "candidate.parquet"
    pq.write_table(clean, clean_path)
    pq.write_table(altered, damaged_path)

    raw_evidence = {
        "reference": {"exists": True, "row_count": 3,
                      "schema": {col: str(clean.schema.field(col).type)
                                 for col in clean.schema.names}},
        "candidate": {"exists": True, "row_count": 3,
                      "schema": {col: str(altered.schema.field(col).type)
                                 for col in altered.schema.names}},
        "schema_diff": {
            "missing_columns": [], "added_columns": [],
            "type_changes": [{"column": "total_amount",
                              "reference_type": "double", "candidate_type": "string"}],
        },
    }
    f2 = run_b0_workflow(
        incident_id="demo-f2-type-change",
        features={**CLEAN_FEATURES, "type_change_count": 1, "revenue_probe_failed": 1},
        raw_evidence=raw_evidence,
        candidate_parquet=damaged_path,
        clean_reference_parquet=clean_path,
        output_dir=run,
    )
    f3 = run_b0_workflow(
        incident_id="demo-f3-risky-quarantine",
        features={**CLEAN_FEATURES, "max_null_rate_increase_pp": 30.0},
        output_dir=run,
    )
    print("F2 diagnosis:", f2["diagnosis"]["failure_type"])
    print("F2 repair proposal:", f2["proposal"]["action"],
          "|", f2["proposal"]["status"])
    print("F2 sandbox:", f2["sandbox"]["status"])
    print("F2 validation:", f2["validation"]["status"])
    print("F2 review:", f2["review_status"], "(not auto-approved)")
    print("F3 diagnosis:", f3["diagnosis"]["failure_type"])
    print("F3 repair proposal:", f3["proposal"]["action"],
          "|", f3["proposal"]["status"], "(quarantine NOT executed)")
    print("Live data modified:", f2["live_data_modified"] or f3["live_data_modified"])
    print("Audit records:", run)


if __name__ == "__main__":
    main()
