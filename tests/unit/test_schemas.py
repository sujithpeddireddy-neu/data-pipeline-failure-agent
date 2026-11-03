import pytest

from pipefail.contracts.schemas import (
    IncidentEvidence,
    DetectionResult,
    DiagnosisResult,
    FailureType,
    RepairAction,
)


def test_failure_types():
    """Verify that the six failure categories exist."""

    assert FailureType.F1.value == "F1"
    assert FailureType.F2.value == "F2"
    assert FailureType.F3.value == "F3"
    assert FailureType.F4.value == "F4"
    assert FailureType.F5.value == "F5"
    assert FailureType.F6.value == "F6"


def test_incident_evidence():
    """Verify that incident evidence can be created."""

    incident = IncidentEvidence(
        incident_id="INC-001",
        pipeline_run_id="RUN-001",
        month="2023-01",
        logs=["Unexpected null values"],
    )

    assert incident.incident_id == "INC-001"
    assert len(incident.logs) == 1
    assert incident.sample_rows == []


def test_sample_row_limit():
    """Verify that incident evidence cannot exceed 20 sample rows."""

    with pytest.raises(ValueError):
        IncidentEvidence(
            incident_id="INC-002",
            pipeline_run_id="RUN-002",
            month="2023-01",
            sample_rows=[{} for _ in range(21)],
        )


def test_detection_result():
    """Verify that detection results use the shared interface."""

    result = DetectionResult(
        incident_id="INC-001",
        model_name="isolation_forest",
        is_suspicious=True,
        confidence=0.95,
    )

    assert result.is_suspicious is True
    assert result.confidence == 0.95


def test_diagnosis_result():
    """Verify that a diagnosis can contain a permitted repair."""

    result = DiagnosisResult(
        incident_id="INC-001",
        model_name="lightgbm",
        failure_type=FailureType.F3,
        affected_column="rate_code",
        recommended_action=RepairAction.MANUAL_REVIEW,
        confidence=0.85,
    )

    assert result.failure_type == FailureType.F3
    assert result.recommended_action == RepairAction.MANUAL_REVIEW


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_invalid_confidence(confidence):
    """Verify that invalid diagnosis confidence values are rejected."""

    with pytest.raises(ValueError):
        DiagnosisResult(
            incident_id="INC-001",
            model_name="lightgbm",
            failure_type=FailureType.F3,
            confidence=confidence,
        )


def test_unknown_failure():
    """Verify that the system supports uncertain diagnoses."""

    result = DiagnosisResult(
        incident_id="INC-003",
        model_name="claude",
        failure_type=FailureType.UNKNOWN,
    )

    assert result.failure_type == FailureType.UNKNOWN
    assert result.recommended_action == RepairAction.MANUAL_REVIEW