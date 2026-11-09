"""B0 F2 training-case pilot using a frozen pre-injection raw schema contract.

The clean baseline is passed ONLY to the validator, never to the repair
contract constructor. This is for training-period debugging, not an unbiased
held-out success estimate.
"""

import argparse
import json
import uuid
from pathlib import Path

from pipefail.contracts.schema_registry import load_contract
from pipefail.workflows.orchestrator import run_b0_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/b0_training_contract_pilot"))
    args = parser.parse_args()
    folder = args.experiment_dir
    contract = load_contract(args.contract)

    # This runner cannot be used on 2023+ validation/test incidents.
    if folder.parent.parent.name != contract.month or "-train-" not in folder.name:
        raise ValueError("Training-only pilot: contract month must match the training experiment")

    features = json.loads((folder / "features.json").read_text(encoding="utf-8"))["features"]
    evidence = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))

    # Check the proposed target type against an independently frozen source.
    changes = evidence.get("schema_diff", {}).get("type_changes", [])
    if not isinstance(changes, list) or len(changes) != 1:
        raise ValueError("F2 pilot requires exactly one documented type change")
    ref_schema = evidence["reference"]["schema"]
    candidate_schema = evidence["candidate"]["schema"]
    changed_cols = [name for name in ref_schema if ref_schema[name] != candidate_schema.get(name)]
    if len(changed_cols) != 1:
        raise ValueError("F2 pilot requires exactly one mismatched schema column")
    column = changed_cols[0]
    expected_type = ref_schema[column]
    nullable = contract.nullable_for(column, expected_type)

    record = run_b0_workflow(
        incident_id="f2-train-" + uuid.uuid4().hex[:8],
        features=features,
        raw_evidence=evidence,
        candidate_parquet=folder / "injected.parquet",
        clean_reference_parquet=folder / "baseline.parquet",  # validation ONLY
        output_dir=args.output_dir / ("pilot-" + uuid.uuid4().hex[:8]),
        trusted_nullability={column: nullable},
        schema_contract_id=contract.contract_id,
    )
    print("Contract:", contract.contract_id)
    print("Original raw source SHA256:", contract.source_sha256)
    print("Contract field:", column, expected_type, "nullable=", nullable)
    print("Detected:", record["detection"]["is_suspicious"])
    print("Diagnosis:", record["diagnosis"]["failure_type"])
    print("Sandbox:", record["sandbox"]["status"] if record["sandbox"] else "NOT_EXECUTED")
    print("Validation:", record["validation"]["status"] if record["validation"] else "NOT_APPLICABLE")
    print("Review:", record["review_status"])
    print("Live data modified:", record["live_data_modified"])
    print("NOTE: training-case debugging only; not held-out test performance")


if __name__ == "__main__":
    main()
