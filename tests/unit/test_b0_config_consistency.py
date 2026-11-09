import json

import pytest

from pipefail.detection.rules import (
    RuleThresholds,
    load_rule_thresholds,
)


def test_python_defaults_match_committed_configuration():
    defaults = RuleThresholds()
    configured = load_rule_thresholds()

    assert defaults == configured
    assert configured.null_spike_min_pp == 3.0


def test_missing_configuration_raises_error(tmp_path):
    missing = tmp_path / "missing_rules.json"

    with pytest.raises(FileNotFoundError):
        load_rule_thresholds(missing)


def test_custom_configuration_is_loaded(tmp_path):
    config = vars(RuleThresholds()).copy()
    config["null_spike_min_pp"] = 6.0

    path = tmp_path / "custom_rules.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_rule_thresholds(path)

    assert loaded.null_spike_min_pp == 6.0
    assert loaded.row_loss_min_pct == config["row_loss_min_pct"]
