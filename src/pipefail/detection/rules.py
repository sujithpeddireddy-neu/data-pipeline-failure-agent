"""B0 deterministic detection rules for the existing 16-feature evidence schema.

A detection decision is *not* a diagnosis or permission to modify live data.
Thresholds are candidate values to tune using training/validation examples only.
"""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Mapping

from pipefail.contracts.schemas import DetectionResult, FailureType


FEATURE_NAMES = (
    "file_missing",
    "schema_comparison_available",
    "missing_column_count",
    "added_column_count",
    "type_change_count",
    "row_count_available",
    "row_count_change_pct",
    "max_null_rate_increase_pp",
    "max_null_rate_decrease_pp",
    "invalid_rate_available",
    "invalid_distance_increase_pp",
    "distance_probe_failed",
    "revenue_probe_failed",
    "arrival_available",
    "late_arrival_minutes",
    "late_arrival_flag",
)


@dataclass(frozen=True)
class RuleThresholds:
    """B0 thresholds; percentage-point fields are NOT percentages of a baseline."""

    null_spike_min_pp: float = 10.0
    row_loss_min_pct: float = 10.0
    invalid_distance_min_pp: float = 5.0
    unexpected_row_gain_min_pct: float = 50.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{item.name} must be a positive finite number")


@dataclass(frozen=True)
class RuleEvaluation:
    """Internal, inspectable signals shared by detection and runbook diagnosis."""

    matches: dict[FailureType, str]
    review_reasons: tuple[str, ...]

    @property
    def suspicious(self) -> bool:
        return bool(self.matches or self.review_reasons)

    @property
    def reasons(self) -> list[str]:
        return [*self.matches.values(), *self.review_reasons]


def analyze_features(
    features: Mapping[str, int | float],
    thresholds: RuleThresholds | None = None,
) -> RuleEvaluation:
    """Evaluate a complete feature vector without using ground-truth labels.

    Missing or invalid feature values fail closed (ValueError); a caller must not
    silently interpret an incomplete record as CLEAN.
    """
    if thresholds is None:
        thresholds = RuleThresholds()

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Incomplete evidence features: missing {', '.join(missing)}")

    value: dict[str, float] = {}
    for name in FEATURE_NAMES:
        raw = features[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Feature {name!r} must be numeric, got {raw!r}")
        number = float(raw)
        if not isfinite(number):
            raise ValueError(f"Feature {name!r} must be finite")
        value[name] = number

    for name in (
        "file_missing", "schema_comparison_available", "row_count_available",
        "invalid_rate_available", "distance_probe_failed", "revenue_probe_failed",
        "arrival_available", "late_arrival_flag",
    ):
        if value[name] not in (0.0, 1.0):
            raise ValueError(f"Flag {name!r} must be 0 or 1")

    for name in ("missing_column_count", "added_column_count", "type_change_count"):
        if value[name] < 0 or not value[name].is_integer():
            raise ValueError(f"Count {name!r} must be a nonnegative integer")

    matches: dict[FailureType, str] = {}
    review: list[str] = []

    if value["file_missing"] == 1:
        matches[FailureType.F6] = "F6: candidate data file is missing"
    if value["arrival_available"] == 1 and value["late_arrival_flag"] == 1:
        matches[FailureType.F6] = "F6: candidate data file arrived late or is missing"

    if value["missing_column_count"] > 0:
        matches[FailureType.F1] = (
            f"F1: {int(value['missing_column_count'])} expected column(s) are missing"
        )
    # Added columns alone are *not* automatically treated as F1: a legitimate
    # TLC schema extension (e.g. cbd_congestion_fee) is a normal change.

    if value["type_change_count"] > 0:
        matches[FailureType.F2] = (
            f"F2: {int(value['type_change_count'])} column type(s) changed"
        )

    if value["max_null_rate_increase_pp"] >= thresholds.null_spike_min_pp:
        matches[FailureType.F3] = (
            f"F3: null rate increased by {value['max_null_rate_increase_pp']:.2f} pp "
            f"(threshold {thresholds.null_spike_min_pp:.2f} pp)"
        )

    if (value["row_count_available"] == 1 and
            value["row_count_change_pct"] <= -thresholds.row_loss_min_pct):
        matches[FailureType.F4] = (
            f"F4: row count changed by {value['row_count_change_pct']:.2f}% "
            f"(loss threshold {thresholds.row_loss_min_pct:.2f}%)"
        )

    if (value["invalid_rate_available"] == 1 and
            value["invalid_distance_increase_pp"] >= thresholds.invalid_distance_min_pp):
        matches[FailureType.F5] = (
            f"F5: negative-trip-distance rate increased by "
            f"{value['invalid_distance_increase_pp']:.2f} pp "
            f"(threshold {thresholds.invalid_distance_min_pp:.2f} pp)"
        )

    if value["file_missing"] == 0:
        if value["schema_comparison_available"] == 0:
            review.append("Schema comparison unavailable: manual review required")
        if value["row_count_available"] == 0:
            review.append("Row-count comparison unavailable: manual review required")
        if (value["row_count_available"] == 1 and
                value["row_count_change_pct"] >= thresholds.unexpected_row_gain_min_pct):
            review.append("Unexpected large row-count increase: no supported F1-F6 rule")

    if (value["distance_probe_failed"] == 1 or
            value["revenue_probe_failed"] == 1) and not matches:
        review.append("SQL probe failed without an unambiguous failure signature")

    if value["arrival_available"] == 0 and value["late_arrival_flag"] == 1:
        review.append("Inconsistent arrival evidence: late flag with no arrival record")

    return RuleEvaluation(matches=matches, review_reasons=tuple(review))


def detect_failure(
    features: Mapping[str, int | float],
    incident_id: str,
    thresholds: RuleThresholds | None = None,
) -> DetectionResult:
    """Return DetectionResult; confidence=None because rules are uncalibrated."""
    evaluation = analyze_features(features, thresholds)
    return DetectionResult(
        incident_id=incident_id,
        model_name="B0_RULES",
        is_suspicious=evaluation.suspicious,
        confidence=None,
        evidence=evaluation.reasons if evaluation.suspicious else ["No B0 rule triggered"],
    )
