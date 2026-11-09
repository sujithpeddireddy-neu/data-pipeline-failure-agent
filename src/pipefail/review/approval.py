"""Record an explicit human decision. This module never deploys repairs."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re

from pipefail.repair.actions import RepairProposal
from pipefail.repair.validator import ValidationResult


@dataclass(frozen=True)
class ReviewDecision:
    incident_id: str
    reviewer: str
    decision: str  # APPROVE | REJECT | DEFER
    recorded_at_utc: str
    notes: str
    deployment_performed: bool = False


def record_review(
    proposal: RepairProposal,
    validation: ValidationResult | None,
    *,
    reviewer: str,
    decision: str,
    notes: str,
    log_dir: str | Path,
) -> ReviewDecision:
    """Approve the *proposal record* only; never change live files or DBs."""
    decision = decision.upper()
    if decision not in {"APPROVE", "REJECT", "DEFER"}:
        raise ValueError("Decision must be APPROVE, REJECT, or DEFER")
    if not reviewer.strip():
        raise ValueError("Reviewer identity must be nonempty")
    if decision == "APPROVE" and (
        proposal.status != "SANDBOX_ELIGIBLE"
        or validation is None
        or validation.incident_id != proposal.incident_id
        or not validation.passed
    ):
        raise ValueError("Cannot approve a repair without matching successful validation")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", proposal.incident_id):
        raise ValueError("Unsafe incident ID")
    record = ReviewDecision(
        incident_id=proposal.incident_id,
        reviewer=reviewer.strip(),
        decision=decision,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    destination = path / f"{proposal.incident_id}_review.json"
    # Refuse silent rewrite of an existing human decision.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(asdict(record), stream, indent=2)
    return record
