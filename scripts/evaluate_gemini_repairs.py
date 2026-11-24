"""B1/B2 diagnosis and operational reliability evaluation."""

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean, median

import pandas as pd

from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook
from pipefail.diagnosis.gemini_agent import GeminiAgent
from pipefail.workflows.gemini_workflow import run_gemini_workflow


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL"))
    p.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gemini_evaluation"),
    )

    a = p.parse_args()

    if not a.model or a.limit < 1:
        p.error("Specify Gemini model and positive limit")

    rec = LightGBMRunbook(
        a.model_dir / "model.txt",
        a.model_dir / "run_manifest.json",
    )

    agent = GeminiAgent(a.model)

    df = pd.read_csv(
        a.data_dir / f"{a.split}.csv"
    ).head(a.limit)

    if df.empty:
        p.error("No incidents available for evaluation")

    rows = []
    total = len(df)

    print("=" * 60)
    print("B1 VS B2 EVALUATION")
    print("Gemini model:", a.model)
    print("Dataset:", a.data_dir)
    print("Split:", a.split)
    print("Total incidents:", total)
    print("=" * 60)

    for i, row in enumerate(df.to_dict("records")):

        incident = f"b2-{a.split}-{i:04d}"

        print(
            f"\n[{i + 1}/{total}] Evaluating {incident}",
            flush=True,
        )

        features = {
            k: row[k] for k in rec.feature_names
        }

        b1, _ = rec.recommend(features, incident)

        b1label = (
            "CLEAN"
            if b1.failure_type.value == "NONE"
            else b1.failure_type.value
        )

        result = run_gemini_workflow(
            recommender=rec,
            gemini=agent,
            incident_id=incident,
            features=features,
            output_dir=a.output_dir / "audits",
        )

        b2label = result["diagnosis"]["failure_type"]

        if b2label == "NONE":
            b2label = "CLEAN"

        api = result.get("api_metrics", {})

        attempts = api.get("attempts") or 0
        latency = api.get("latency_seconds")
        request_errors = api.get("request_errors", [])

        deferred = b2label == "UNKNOWN"

        workflow_error = (
            result["review_status"] == "BLOCKED_GEMINI_ERROR"
        )

        api_failure = bool(
            workflow_error
            and request_errors
            and request_errors[-1]["attempt"] == attempts
        )

        response_failure = (
            workflow_error and not api_failure
        )

        record = {
            "incident": incident,
            "true_label": row["label"],
            "b1_label": b1label,
            "b2_label": b2label,
            "b1_correct": b1label == row["label"],
            "b2_correct": b2label == row["label"],
            "b2_deferred": deferred,
            "api_failure": api_failure,
            "response_failure": response_failure,
            "api_attempts": attempts,
            "api_latency_seconds": latency,
            "retry_recovered": api.get(
                "retry_recovered", False
            ),
            "review_status": result["review_status"],
            "repair_proposal": result["proposal"]["action"],
            "live_data_modified": result["live_data_modified"],
        }

        rows.append(record)

        print(
            f"Actual: {row['label']} | "
            f"B1: {b1label} | "
            f"B2: {b2label} | "
            f"Attempts: {attempts} | "
            f"Latency: {latency}s",
            flush=True,
        )

    n = len(rows)

    b1_correct = sum(r["b1_correct"] for r in rows)
    b2_correct = sum(r["b2_correct"] for r in rows)

    deferrals = sum(r["b2_deferred"] for r in rows)
    api_failures = sum(r["api_failure"] for r in rows)

    response_failures = sum(
        r["response_failure"] for r in rows
    )

    non_deferred = [
        r for r in rows if not r["b2_deferred"]
    ]

    retried = sum(
        r["api_attempts"] > 1 for r in rows
    )

    recovered = sum(
        r["retry_recovered"] for r in rows
    )

    latencies = [
        r["api_latency_seconds"]
        for r in rows
        if r["api_latency_seconds"] is not None
    ]

    metrics = {
        "model": a.model,
        "split": a.split,
        "total_incidents": n,

        "b1_accuracy": b1_correct / n,
        "b2_accuracy": b2_correct / n,

        "b2_correct_diagnoses": b2_correct,

        "b2_non_deferred_accuracy": (
            sum(r["b2_correct"] for r in non_deferred)
            / len(non_deferred)
            if non_deferred else None
        ),

        "b2_deferrals": deferrals,
        "b2_deferral_rate": deferrals / n,
        "b2_coverage": len(non_deferred) / n,

        "api_failures": api_failures,
        "api_failure_rate": api_failures / n,

        "response_failures": response_failures,
        "response_failure_rate": response_failures / n,

        "successful_diagnosis_response_rate": (
            n - api_failures - response_failures
        ) / n,

        "total_api_attempts": sum(
            r["api_attempts"] for r in rows
        ),

        "retried_incidents": retried,
        "retry_rate": retried / n,

        "recovered_incidents": recovered,
        "retry_recovery_rate": (
            recovered / retried if retried else None
        ),

        "average_latency_seconds": (
            mean(latencies) if latencies else None
        ),

        "median_latency_seconds": (
            median(latencies) if latencies else None
        ),

        "maximum_latency_seconds": (
            max(latencies) if latencies else None
        ),

        "total_latency_seconds": sum(latencies),

        "live_data_modified_count": sum(
            r["live_data_modified"] for r in rows
        ),

        "repair_success_measured": False,
    }

    a.output_dir.mkdir(parents=True, exist_ok=True)

    with (a.output_dir / "comparison.csv").open(
        "x", newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    with (a.output_dir / "metrics.json").open("x") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL B1 VS B2 RESULTS")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nResults saved to:", a.output_dir)
    print("Repair success not measured.")
    print("Test split may have been previously inspected.")


if __name__ == "__main__":
    main()
