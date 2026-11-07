"""B0 detection + diagnosis tests. Does not depend on the private DVC remote."""

import pytest

from pipefail.contracts.schemas import FailureType, RepairAction
from pipefail.detection.rules import RuleThresholds, detect_failure
from pipefail.diagnosis.runbook import diagnose_failure


BASE = {
    "file_missing": 0,
    "schema_comparison_available": 1,
    "missing_column_count": 0,
    "added_column_count": 0,
    "type_change_count": 0,
    "row_count_available": 1,
    "row_count_change_pct": 0.0,
    "max_null_rate_increase_pp": 0.0,
    "max_null_rate_decrease_pp": 0.0,
    "invalid_rate_available": 1,
    "invalid_distance_increase_pp": 0.0,
    "distance_probe_failed": 0,
    "revenue_probe_failed": 0,
    "arrival_available": 0,
    "late_arrival_minutes": 0.0,
    "late_arrival_flag": 0,
}


def example(**updates):
    return {**BASE, **updates}


def test_clean_and_legitimate_additive_schema_change():
    for features in [example(), example(added_column_count=1)]:
        assert not detect_failure(features, "case").is_suspicious
        diagnosis = diagnose_failure(features, "case")
        assert diagnosis.failure_type == FailureType.NONE
        assert diagnosis.confidence is None


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({"missing_column_count": 1}, FailureType.F1),
        ({"type_change_count": 1}, FailureType.F2),
        ({"max_null_rate_increase_pp": 25.0}, FailureType.F3),
        ({"row_count_change_pct": -30.0}, FailureType.F4),
        ({"invalid_distance_increase_pp": 12.0}, FailureType.F5),
        ({"file_missing": 1, "schema_comparison_available": 0,
          "row_count_available": 0, "invalid_rate_available": 0}, FailureType.F6),
        ({"arrival_available": 1, "late_arrival_flag": 1,
          "late_arrival_minutes": 95.0}, FailureType.F6),
    ],
)
def test_each_failure_rule(updates, expected):
    features = example(**updates)
    assert detect_failure(features, "case").is_suspicious
    diagnosis = diagnose_failure(features, "case")
    assert diagnosis.failure_type == expected
    assert diagnosis.confidence is None


def test_multiple_signals_defer_instead_of_guessing():
    features = example(type_change_count=1, missing_column_count=1)
    diagnosis = diagnose_failure(features, "case")
    assert diagnosis.failure_type == FailureType.UNKNOWN
    assert diagnosis.recommended_action == RepairAction.MANUAL_REVIEW


def test_probe_failure_without_specific_signature_deferred():
    features = example(distance_probe_failed=1)
    assert detect_failure(features, "case").is_suspicious
    assert diagnose_failure(features, "case").failure_type == FailureType.UNKNOWN


def test_incomplete_comparison_deferred():
    features = example(schema_comparison_available=0)
    assert detect_failure(features, "case").is_suspicious
    assert diagnose_failure(features, "case").failure_type == FailureType.UNKNOWN


def test_unexpected_row_gain_needs_review():
    features = example(row_count_change_pct=80.0)
    assert detect_failure(features, "case").is_suspicious
    assert diagnose_failure(features, "case").failure_type == FailureType.UNKNOWN


def test_no_data_changing_actions_from_numeric_features_only():
    assert diagnose_failure(example(type_change_count=1), "case").recommended_action == RepairAction.MANUAL_REVIEW
    assert diagnose_failure(example(max_null_rate_increase_pp=30), "case").recommended_action == RepairAction.MANUAL_REVIEW
    assert diagnose_failure(example(row_count_change_pct=-30), "case").recommended_action == RepairAction.RERUN_MONTH


def test_missing_feature_fails_closed():
    features = example()
    del features["file_missing"]
    with pytest.raises(ValueError, match="Incomplete evidence"):
        detect_failure(features, "case")


def test_bad_flag_fails_closed():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        detect_failure(example(file_missing=2), "case")


def test_thresholds_configurable():
    features = example(max_null_rate_increase_pp=5.0)
    assert not detect_failure(features, "case").is_suspicious
    thresholds = RuleThresholds(null_spike_min_pp=4.0)
    assert diagnose_failure(features, "case", thresholds).failure_type == FailureType.F3
