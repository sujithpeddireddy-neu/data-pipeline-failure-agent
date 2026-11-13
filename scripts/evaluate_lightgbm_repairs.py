"""Generate inspectable LightGBM -> runbook repair suggestions for a benchmark split.

Example (from repo root):
    python scripts/evaluate_lightgbm_repairs.py \
      --data-dir data/benchmark_1200_v1 --split test \
      --model-dir results/walkthrough_v1/lightgbm \
      --output-dir results/walkthrough_v1/lightgbm/repair_recommendations

NO production repairs, NO sandbox execution, NO human approval. For F2,
without full schema evidence, conversion deliberately requires manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmark_1200_v1")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "results/walkthrough_v1/lightgbm")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/walkthrough_v1/lightgbm/repair_recommendations")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Optional uncalibrated score threshold; 0 preserves original predictions")
    args = parser.parse_args()

    input_csv = args.data_dir / f"{args.split}.csv"
    manifest_path = args.model_dir / "run_manifest.json"
    model_path = args.model_dir / "model.txt"
    recommender = LightGBMRunbook(model_path, manifest_path, args.min_confidence)

    expected_sha = recommender.manifest.get("csv_sha256", {}).get(args.split)
    actual_sha = sha256(input_csv)
    if expected_sha and actual_sha.lower() != expected_sha.lower():
        raise ValueError(f"Dataset hash mismatch for {args.split}: model was trained/evaluated on different data")

    data = pd.read_csv(input_csv)
    expected_columns = ["label", *recommender.feature_names]
    if list(data.columns) != expected_columns:
        raise ValueError("Benchmark schema/order differs from recorded model inputs")
    if not data["label"].isin(recommender.label_order).all():
        raise ValueError("Unknown ground-truth label in benchmark")

    rows = []
    for i, record in enumerate(data.to_dict(orient="records"), start=1):
        incident_id = f"{args.split}-{i:04d}"
        diagnosis, proposal = recommender.recommend(record, incident_id)
        rows.append({
            "incident_id": incident_id,
            "true_label": record["label"],
            "predicted_label": diagnosis.failure_type.value if diagnosis.failure_type.value != "NONE" else "CLEAN",
            "diagnosis_correct": (record["label"] == ("CLEAN" if diagnosis.failure_type.value == "NONE" else diagnosis.failure_type.value)),
            "score_uncalibrated": round(diagnosis.confidence, 8),
            "runbook_action": diagnosis.recommended_action.value,
            "proposed_action": proposal.action.value,
            "proposal_status": proposal.status,
            "human_approval_required": proposal.human_approval_required,
            "rationale": " | ".join(proposal.rationale),
        })

    output = pd.DataFrame(rows)
    original_predictions = args.model_dir / f"{args.split}_predictions.csv"
    if original_predictions.is_file() and args.min_confidence == 0.0:
        original = pd.read_csv(original_predictions)
        if (len(original) != len(output) or
                not original["predicted_label"].astype(str).reset_index(drop=True).equals(
                    output["predicted_label"].astype(str).reset_index(drop=True))):
            raise ValueError("Current LightGBM predictions differ from original baseline predictions")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.split}_repair_suggestions.csv"
    summary_path = args.output_dir / f"{args.split}_repair_summary.json"
    output.to_csv(output_path, index=False)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    report = {
        "split": args.split,
        "n_incidents": len(output),
        "diagnosis_correct": int(output["diagnosis_correct"].sum()),
        "diagnosis_accuracy": float(output["diagnosis_correct"].mean()),
        "predicted_labels": dict(Counter(output["predicted_label"])),
        "runbook_actions": dict(Counter(output["runbook_action"])),
        "proposal_statuses": dict(Counter(output["proposal_status"])),
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "model_training_commit": recommender.manifest.get("git_commit"),
        "model_sha256": sha256(model_path),
        "model_manifest_sha256": sha256(manifest_path),
        "dataset_csv_sha256": actual_sha,
        "min_confidence_uncalibrated": args.min_confidence,
        "live_data_modified": False,
        "sandbox_repair_trials": 0,
        "successful_repairs": "NOT_EVALUATED",
        "harmful_repairs": "NOT_EVALUATED",
        "limitations": [
            "This output is diagnosis plus a *proposal*, not successful repair execution.",
            "Numeric evidence alone cannot identify a safely repairable affected column.",
            "F2 type conversion requires full trusted schema evidence and a separate sandbox trial.",
            "The LightGBM score is uncalibrated and does not establish repair safety.",
            "The benchmark contains synthetic failures and repeated feature vectors across splits.",
        ],
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Split: {args.split} | incidents: {len(output)}")
    print(f"Diagnoses correct: {report['diagnosis_correct']}/{len(output)}")
    print("Runbook actions:", report["runbook_actions"])
    print("Proposal statuses:", report["proposal_statuses"])
    print("No changes to source files; no repair execution or approval.")
    print("Saved:", output_path)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
