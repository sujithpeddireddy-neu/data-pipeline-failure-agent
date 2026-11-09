"""B0 runbook diagnosis and *recommendations* (no live repair execution).

Ambiguous or incomplete evidence defers to a human. Numeric features cannot
identify an affected column, so no blind casts, renames or quarantines are made.
"""

from typing import Mapping

from pipefail.contracts.schemas import (
    DiagnosisResult, FailureType, IncidentEvidence, RepairAction,
)
from pipefail.detection.rules import RuleThresholds, analyze_features


# These are proposals only. Every live change requires later validation and
# approval; MANUAL_REVIEW intentionally avoids risky guessed transformations.
DEFAULT_ACTIONS: dict[FailureType, RepairAction] = {
    FailureType.F1: RepairAction.PAUSE_DOWNSTREAM,
    FailureType.F2: RepairAction.MANUAL_REVIEW,
    FailureType.F3: RepairAction.MANUAL_REVIEW,
    FailureType.F4: RepairAction.RERUN_MONTH,
    FailureType.F5: RepairAction.MANUAL_REVIEW,
    FailureType.F6: RepairAction.PAUSE_DOWNSTREAM,
}


def _affected_column(
    failure_type: FailureType,
    evidence: IncidentEvidence | None,
) -> str | None:
    if evidence is None:
        return None
    diff = evidence.schema_diff
    if failure_type == FailureType.F1:
        missing = diff.get("missing_columns", [])
        if isinstance(missing, list) and len(missing) == 1 and isinstance(missing[0], str):
            return missing[0]
    if failure_type == FailureType.F2:
        changes = diff.get("type_changes", [])
        if isinstance(changes, list) and len(changes) == 1 and isinstance(changes[0], dict):
            for name in ("column", "column_name", "name"):
                if isinstance(changes[0].get(name), str):
                    return changes[0][name]
    return None


def diagnose_failure(
    features: Mapping[str, int | float],
    incident_id: str,
    thresholds: RuleThresholds | None = None,
    evidence: IncidentEvidence | None = None,
) -> DiagnosisResult:
    """Classify the incident conservatively and return a typed recommendation."""
    evaluation = analyze_features(features, thresholds)
    hits = list(evaluation.matches)

    if len(hits) > 1 or evaluation.review_reasons:
        return DiagnosisResult(
            incident_id=incident_id,
            model_name="B0_RUNBOOK",
            failure_type=FailureType.UNKNOWN,
            recommended_action=RepairAction.MANUAL_REVIEW,
            evidence=evaluation.reasons,
            confidence=None,
            uncertainty="Conflicting/insufficient signals: defer to on-call engineer",
        )

    if not hits:
        return DiagnosisResult(
            incident_id=incident_id,
            model_name="B0_RUNBOOK",
            failure_type=FailureType.NONE,
            recommended_action=RepairAction.MANUAL_REVIEW,  # contract has no NO_ACTION
            evidence=["No B0 failure rule triggered; no repair recommended"],
            confidence=None,
        )

    failure_type = hits[0]
    action = DEFAULT_ACTIONS[failure_type]
    return DiagnosisResult(
        incident_id=incident_id,
        model_name="B0_RUNBOOK",
        failure_type=failure_type,
        affected_column=_affected_column(failure_type, evidence),
        recommended_action=action,
        evidence=evaluation.reasons,
        confidence=None,
        uncertainty=(
            "Manual investigation required before choosing a data-changing action"
            if action == RepairAction.MANUAL_REVIEW else
            "Recommendation only; requires validation and human approval"
        ),
    )
