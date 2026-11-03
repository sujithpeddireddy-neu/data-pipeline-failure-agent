"""
Shared data contracts for the Data Pipeline Failure Agent.

These schemas define how pipeline evidence, detection results,
and diagnosis results are exchanged between system components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    """Supported pipeline failure categories."""

    F1 = "F1"  # Schema change
    F2 = "F2"  # Data type change
    F3 = "F3"  # Null spike
    F4 = "F4"  # Row-count shift
    F5 = "F5"  # Invalid values
    F6 = "F6"  # Missing or late file

    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class RepairAction(str, Enum):
    """Allowed repair recommendations."""

    RENAME_COLUMN = "RENAME_COLUMN"
    CONVERT_TYPE = "CONVERT_TYPE"
    QUARANTINE_ROWS = "QUARANTINE_ROWS"
    RERUN_MONTH = "RERUN_MONTH"
    PAUSE_DOWNSTREAM = "PAUSE_DOWNSTREAM"
    UPDATE_SQL = "UPDATE_SQL"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass
class IncidentEvidence:
    """Evidence collected from a pipeline execution."""

    incident_id: str
    pipeline_run_id: str
    month: str

    logs: list[str] = field(default_factory=list)
    schema_diff: dict[str, Any] = field(default_factory=dict)
    column_statistics: dict[str, Any] = field(default_factory=dict)
    code_context: dict[str, str] = field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.sample_rows) > 20:
            raise ValueError(
                "Incident evidence cannot contain more than 20 sample rows."
            )


@dataclass
class DetectionResult:
    """Standardized output from a failure detection model."""

    incident_id: str
    model_name: str
    is_suspicious: bool
    confidence: float | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Standardized output from a diagnosis architecture."""

    incident_id: str
    model_name: str
    failure_type: FailureType

    affected_step: str | None = None
    affected_column: str | None = None

    recommended_action: RepairAction = RepairAction.MANUAL_REVIEW

    evidence: list[str] = field(default_factory=list)
    confidence: float | None = None
    uncertainty: str | None = None

    def __post_init__(self):
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError("Confidence must be between 0 and 1.")