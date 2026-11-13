"""Audited LightGBM -> runbook -> optional *sandbox-only* validation workflow.

No live-data edits and no automatic approval or deployment.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Mapping

from pipefail.contracts.schemas import FailureType
from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook
from pipefail.repair.sandbox import execute_in_sandbox
from pipefail.repair.validator import validate_sandbox


def run_lightgbm_workflow(
    *,
    recommender: LightGBMRunbook,
    incident_id: str,
    features: Mapping[str, int | float],
    output_dir: str | Path,
    raw_evidence: Mapping[str, Any] | None = None,
    candidate_parquet: str | Path | None = None,
    clean_reference_parquet: str | Path | None = None,
    trusted_nullability: Mapping[str, bool] | None = None,
    schema_contract_id: str | None = None,
) -> dict:
    """Produce an append-only audit record and, if eligible, sandbox trial.

    Trusted schema constraints must come from a versioned contract, NOT from
    the held-out clean-reference Parquet file used to validate a trial.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", incident_id):
        raise ValueError("Incident ID must contain only letters, numbers, _ or -")
    if trusted_nullability is not None:
        if not schema_contract_id or not schema_contract_id.strip():
            raise ValueError("Trusted nullability requires schema_contract_id")
        if any(type(value) is not bool for value in trusted_nullability.values()):
            raise ValueError("Trusted nullability values must be booleans")
    output = Path(output_dir)
    record_path = output / f"{incident_id}_audit.json"
    if record_path.exists() or record_path.is_symlink():
        raise FileExistsError(f"Existing audit record: {record_path}")

    diagnosis, proposal = recommender.recommend(features, incident_id, raw_evidence)
    trial = None
    validation = None
    contract_nullable = None
    if proposal.status == "SANDBOX_ELIGIBLE" and candidate_parquet is not None:
        col = proposal.parameters["column"]
        if trusted_nullability is not None:
            contract_nullable = trusted_nullability.get(col)
        trial = execute_in_sandbox(
            proposal, candidate_parquet, output / "sandbox",
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

    if trial is not None and trial.status == "FAILED":
        status = "BLOCKED_SANDBOX_FAILED"
    elif validation is not None and validation.status == "FAILED":
        status = "BLOCKED_VALIDATION_FAILED"
    elif validation is not None and not validation.passed:
        status = "BLOCKED_NOT_VALIDATED"
    elif diagnosis.failure_type == FailureType.NONE:
        status = "NO_ACTION"
    else:
        status = "PENDING_HUMAN_REVIEW"

    audit = {
        "incident_id": incident_id,
        "architecture": "LIGHTGBM_RUNBOOK",
        "diagnosis": {
            **asdict(diagnosis),
            "failure_type": diagnosis.failure_type.value,
            "recommended_action": diagnosis.recommended_action.value,
        },
        "proposal": {
            **asdict(proposal),
            "failure_type": proposal.failure_type.value,
            "action": proposal.action.value,
        },
        "sandbox": asdict(trial) if trial is not None else None,
        "validation": asdict(validation) if validation is not None else None,
        "review_status": status,
        "schema_contract": (
            {"id": schema_contract_id, "target_column_nullable": contract_nullable}
            if contract_nullable is not None else None
        ),
        "live_data_modified": False,
        "limitations": [
            "Only one isolated incident assessed; no production deployment or automatic approval.",
            "Classification confidence is uncalibrated.",
            "A sandbox validation pass on a small sample is not production-wide repair safety.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2)
    return audit
