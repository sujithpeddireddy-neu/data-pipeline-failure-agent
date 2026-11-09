"""B0 constrained repair proposals. Recommendations are NEVER applied to live data."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from pipefail.contracts.schemas import DiagnosisResult, FailureType, RepairAction


# Only a type restoration, checked against an independent clean reference, is
# eligible for automatic sandbox experimentation in B0 v1. Everything else
# deliberately defers; absence of an action is not a successfully repaired case.
RESTORABLE_TYPES = frozenset({"double", "float", "int64", "int32"})


@dataclass(frozen=True)
class RepairProposal:
    incident_id: str
    failure_type: FailureType
    action: RepairAction
    status: str   # NO_ACTION | REVIEW_REQUIRED | SANDBOX_ELIGIBLE
    rationale: tuple[str, ...] = ()
    parameters: dict[str, str] = field(default_factory=dict)
    human_approval_required: bool = True


def _review(diagnosis: DiagnosisResult, *reasons: str) -> RepairProposal:
    return RepairProposal(
        incident_id=diagnosis.incident_id,
        failure_type=diagnosis.failure_type,
        action=diagnosis.recommended_action,
        status="REVIEW_REQUIRED",
        rationale=tuple(reasons),
    )


def propose_repair(
    diagnosis: DiagnosisResult,
    raw_evidence: Mapping[str, Any] | None = None,
) -> RepairProposal:
    """Return a proposal; never guess a column or cast from numeric features.

    F2 requires raw evidence containing full reference/candidate schemas and a
    single documented type change. Those facts make only a sandbox TRIAL valid,
    not a production repair. Other failure types get the runbook action for review.
    """
    if diagnosis.failure_type is FailureType.NONE:
        return RepairProposal(
            incident_id=diagnosis.incident_id,
            failure_type=diagnosis.failure_type,
            action=RepairAction.MANUAL_REVIEW,  # existing contract has no NO_ACTION
            status="NO_ACTION",
            rationale=("Run assessed as clean; no repair needed.",),
        )
    if diagnosis.failure_type is not FailureType.F2:
        return _review(
            diagnosis,
            "No verified automatic data-changing repair for this failure class.",
            "Human operator must investigate or take the runbook's operational action.",
        )
    if not isinstance(raw_evidence, Mapping):
        return _review(diagnosis, "F2 type restoration requires detailed raw evidence.")

    reference = raw_evidence.get("reference", {})
    candidate = raw_evidence.get("candidate", {})
    diff = raw_evidence.get("schema_diff", {})
    if not all(isinstance(part, Mapping) for part in (reference, candidate, diff)):
        return _review(diagnosis, "Malformed reference/candidate schema evidence.")
    reference_schema = reference.get("schema")
    candidate_schema = candidate.get("schema")
    changes = diff.get("type_changes")
    if (not isinstance(reference_schema, dict)
            or not isinstance(candidate_schema, dict)
            or not isinstance(changes, list)
            or len(changes) != 1
            or not isinstance(changes[0], dict)
            or diff.get("missing_columns", []) != []
            or diff.get("added_columns", []) != []):
        return _review(diagnosis, "Schema evidence does not identify one isolated type change.")
    if reference_schema.keys() != candidate_schema.keys():
        return _review(diagnosis, "Reference and candidate columns differ.")
    different = [key for key in reference_schema if reference_schema[key] != candidate_schema[key]]
    if len(different) != 1:
        return _review(diagnosis, "Exactly one changed column is required.")
    column = different[0]
    changed = changes[0]
    # Collector's change objects may use different name fields; do not trust a
    # mismatching name from a manifest or treat a missing one as authoritative.
    documented = next((changed[k] for k in ("column", "column_name", "name")
                       if k in changed), None)
    if documented is not None and documented != column:
        return _review(diagnosis, "Column disagreement between schema and diff.")
    if diagnosis.affected_column is not None and diagnosis.affected_column != column:
        return _review(diagnosis, "Column disagreement between diagnosis and schema.")
    target = reference_schema[column]
    observed = candidate_schema[column]
    if target not in RESTORABLE_TYPES or not isinstance(observed, str):
        return _review(diagnosis, "Restoration target type is not allowlisted.")
    # Refuse to restore an undiagnosed type, missing input file or row-count drift.
    if candidate.get("exists") is False:
        return _review(diagnosis, "Candidate file is missing.")
    if (reference.get("row_count") is not None and candidate.get("row_count") is not None
            and reference["row_count"] != candidate["row_count"]):
        return _review(diagnosis, "Row counts differ before repair.")

    return RepairProposal(
        incident_id=diagnosis.incident_id,
        failure_type=diagnosis.failure_type,
        action=RepairAction.CONVERT_TYPE,
        status="SANDBOX_ELIGIBLE",
        rationale=(f"Trial restore {column} from {observed} to {target} in sandbox only.",
                   "Pass/fail requires exact comparison against the clean reference.",
                   "Human approval is mandatory; approval does not deploy changes."),
        parameters={"column": column, "source_type": observed, "target_type": target},
    )
