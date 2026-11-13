"""LightGBM diagnosis -> existing conservative B0 repair runbook.

Only loads an already-trained LightGBM model. Predicts labels using the saved
training-time feature order and class order; does not fit, repair, deploy, or
approve anything. The reported LightGBM confidence is NOT calibrated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipefail.contracts.schemas import DiagnosisResult, FailureType, RepairAction
from pipefail.diagnosis.runbook import DEFAULT_ACTIONS
from pipefail.repair.actions import RepairProposal, propose_repair


class LightGBMRunbook:
    """A frozen, trained diagnosis model combined with safe repair proposals."""

    def __init__(
        self,
        model_path: str | Path,
        manifest_path: str | Path,
        min_confidence: float = 0.0,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        self.model_path = Path(model_path)
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.feature_names = list(self.manifest["features"])
        self.label_order = list(self.manifest["label_order"])
        self.min_confidence = min_confidence

        if len(self.feature_names) != 16 or len(set(self.feature_names)) != 16:
            raise ValueError("Expected the 16 uniquely named PipeFail evidence features")
        if self.label_order != ["CLEAN", "F1", "F2", "F3", "F4", "F5", "F6"]:
            raise ValueError("Unexpected training-time class order")
        self.model = lgb.Booster(model_file=str(self.model_path))
        if self.model.feature_name() != self.feature_names:
            raise ValueError("Model feature names/order differ from training manifest")
        if self.model.num_model_per_iteration() != len(self.label_order):
            raise ValueError("Model class count differs from training manifest")

    def recommend(
        self,
        features: Mapping[str, Any],
        incident_id: str,
        raw_evidence: Mapping[str, Any] | None = None,
    ) -> tuple[DiagnosisResult, RepairProposal]:
        """One evidence vector -> typed diagnosis -> *non-deploying* proposal.

        Passing full raw evidence can make a single F2 conversion eligible for
        a sandbox trial. It never authorizes executing a repair on live data.
        """
        values = []
        for feature in self.feature_names:
            if feature not in features:
                raise ValueError(f"Missing model feature: {feature}")
            value = float(features[feature])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite model feature: {feature}")
            values.append(value)
        ordered = pd.DataFrame([values], columns=self.feature_names)
        probabilities = np.asarray(self.model.predict(ordered), dtype=float)
        if probabilities.shape != (1, len(self.label_order)):
            raise ValueError(f"Unexpected LightGBM prediction shape: {probabilities.shape}")
        if not np.isfinite(probabilities).all():
            raise ValueError("Non-finite model probabilities")
        position = int(np.argmax(probabilities[0]))
        score = float(probabilities[0, position])
        predicted_label = self.label_order[position]

        if score < self.min_confidence:
            failure_type = FailureType.UNKNOWN
            action = RepairAction.MANUAL_REVIEW
            note = f"Below optional {self.min_confidence:.2f} score threshold; human diagnosis required"
        elif predicted_label == "CLEAN":
            failure_type = FailureType.NONE
            action = RepairAction.MANUAL_REVIEW  # schema has no NO_ACTION enum
            note = "Model classified evidence as CLEAN; no repair suggested"
        else:
            failure_type = FailureType(predicted_label)
            action = DEFAULT_ACTIONS[failure_type]
            note = "Rulebook mapping is a suggestion; human review required"

        diagnosis = DiagnosisResult(
            incident_id=incident_id,
            model_name="LIGHTGBM_RUNBOOK",
            failure_type=failure_type,
            recommended_action=action,
            confidence=score,
            evidence=[
                f"LightGBM predicted {predicted_label} from 16 evidence features",
                "This prediction score is not calibrated",
            ],
            uncertainty=note,
        )
        proposal = propose_repair(diagnosis, raw_evidence)
        return diagnosis, proposal
