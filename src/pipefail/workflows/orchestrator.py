"""B0 sample-only workflow: evidence -> detection -> diagnosis -> proposal -> sandbox -> checks.

Never applies a repair to the original data and NEVER auto-approves a change.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import json
import re

from pipefail.detection.rules import RuleThresholds, detect_failure, load_rule_thresholds
from pipefail.diagnosis.runbook import diagnose_failure
from pipefail.repair.actions import propose_repair
from pipefail.repair.sandbox import execute_in_sandbox
from pipefail.repair.validator import validate_sandbox


def run_b0_workflow(
    *,
    incident_id: str,
    features: Mapping[str, int | float],
    output_dir: str | Path,
    raw_evidence: Mapping[str, Any] | None = None,
    candidate_parquet: str | Path | None = None,
    clean_reference_parquet: str | Path | None = None,
    thresholds: RuleThresholds | None = None,
    trusted_nullability: Mapping[str, bool] | None = None,
    schema_contract_id: str | None = None,
) -> dict:
    """Return and save a fully inspectable incident record, but NO approval.

    For most F1-F6 scenarios the conservative baseline deliberately defers.
    The one B0 sandbox trial (F2) requires raw schema evidence and small local
    Parquet files. No reference == no validation pass.
    """
    # Provenance is required for any declared contract. This cannot be
    # populated from the held-out clean reference used by the validator.
    if trusted_nullability is not None:
        if not schema_contract_id or not schema_contract_id.strip():
            raise ValueError("A trusted nullability override requires schema_contract_id")
        if any(type(value) is not bool for value in trusted_nullability.values()):
            raise ValueError("Trusted schema nullability values must be booleans")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", incident_id):
        raise ValueError("Incident ID must contain only letters, numbers, _ or -")
    root = Path(output_dir)
    record_path = root / f"{incident_id}_audit.json"
    if record_path.exists() or record_path.is_symlink():
        raise FileExistsError(f"Existing audit record: {record_path}")

    if thresholds is None:
        thresholds = load_rule_thresholds()
    detection = detect_failure(features, incident_id, thresholds)
    diagnosis = diagnose_failure(features, incident_id, thresholds)
    proposal = propose_repair(diagnosis, raw_evidence)
    trial = None
    validation = None
    contract_nullable = None
    if proposal.status == "SANDBOX_ELIGIBLE" and candidate_parquet is not None:
        col = proposal.parameters.get("column")
        if trusted_nullability is not None:
            contract_nullable = trusted_nullability.get(col)
        trial = execute_in_sandbox(
            proposal, candidate_parquet, root / "sandbox",
            expected_nullable=contract_nullable,
        )
        contract_field = (
            {"name": col, "type": proposal.parameters["target_type"],
             "nullable": contract_nullable}
            if contract_nullable is not None else None
        )
        validation = validate_sandbox(
            trial, clean_reference_parquet,
            expected_restored_field=contract_field,
            schema_contract_id=schema_contract_id if contract_field else None,
        )

    if validation is not None and validation.status == "FAILED":
        review_status = "BLOCKED_VALIDATION_FAILED"
    elif trial is not None and trial.status == "FAILED":
        review_status = "BLOCKED_SANDBOX_FAILED"
    elif validation is not None and not validation.passed:
        review_status = "BLOCKED_NOT_VALIDATED"
    else:
        review_status = "PENDING_HUMAN_REVIEW" if detection.is_suspicious else "NO_ACTION"

    record = {
        "incident_id": incident_id,
        "architecture": "B0_RULES_RUNBOOK",
        "detection": asdict(detection),
        "diagnosis": {**asdict(diagnosis), "failure_type": diagnosis.failure_type.value,
                       "recommended_action": diagnosis.recommended_action.value},
        "proposal": {**asdict(proposal), "failure_type": proposal.failure_type.value,
                     "action": proposal.action.value},
        "sandbox": asdict(trial) if trial else None,
        "validation": asdict(validation) if validation else None,
        "review_status": review_status,
        "schema_contract": (
            {"id": schema_contract_id,
             "target_column_nullable": contract_nullable}
            if contract_nullable is not None else None
        ),
        "live_data_modified": False,
        "limitations": ["Only isolated sample tested; no production deployment or Airflow integration.",
                        "End-to-end repair success and harmful-fix rates are not yet established."],
    }
    root.mkdir(parents=True, exist_ok=True)
    # Append-only per incident; refuse accidental rewrites.
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return record
