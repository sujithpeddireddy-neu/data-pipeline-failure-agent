"""Train and evaluate the PipeFail 7-class LightGBM diagnosis baseline.

Runs with the 1,200-experiment benchmark or the 70-row smoke test.

Examples (from repository root):
    python scripts/train_lightgbm_baseline.py
    python scripts/train_lightgbm_baseline.py --data-dir data/smoke_test --output-dir results/lightgbm_smoke

Uses the training split to fit, validation split only for early stopping,
and test split only for final reporting. No test-set-driven model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 6170
LABELS = ("CLEAN", "F1", "F2", "F3", "F4", "F5", "F6")
FEATURES = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_command(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run 'dvc pull' with access to the configured remote."
        )
    df = pd.read_csv(path)
    expected = ["label", *FEATURES]
    if list(df.columns) != expected:
        raise ValueError(
            f"Unexpected feature schema in {path}.\n"
            f"Expected: {expected}\nReceived: {list(df.columns)}"
        )
    unknown_labels = set(df["label"].astype(str)) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown labels in {path}: {sorted(unknown_labels)}")
    if not df["label"].notna().all():
        raise ValueError(f"Missing labels in {path}")
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES):
        raise ValueError(f"Non-numeric feature columns in {path}")
    if not np.isfinite(df[list(FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite feature values in {path}")
    return df


def feature_fingerprints(df: pd.DataFrame) -> set[int]:
    return set(pd.util.hash_pandas_object(df[list(FEATURES)], index=False).tolist())


def overlap_details(train: pd.DataFrame, other: pd.DataFrame) -> dict:
    train_hashes = feature_fingerprints(train)
    other_hashes = pd.util.hash_pandas_object(other[list(FEATURES)], index=False)
    return {
        "shared_unique_feature_vectors_with_train": len(train_hashes.intersection(set(other_hashes.tolist()))),
        "rows_with_feature_vector_seen_in_train": int(other_hashes.isin(train_hashes).sum()),
        "total_rows": len(other),
    }


def score_split(name: str, df: pd.DataFrame, model: lgb.LGBMClassifier, out_dir: Path) -> dict:
    X = df[list(FEATURES)]
    truth = df["label"].to_numpy()
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)

    class_order = list(model.classes_)
    probability_df = pd.DataFrame(probabilities, columns=[f"prob_{label}" for label in class_order])
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "true_label": truth,
                    "predicted_label": predicted,
                    "confidence": probabilities.max(axis=1),
                }
            ),
            probability_df,
        ],
        axis=1,
    )
    predictions.to_csv(out_dir / f"{name}_predictions.csv", index=False)

    matrix = pd.DataFrame(
        confusion_matrix(truth, predicted, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    )
    matrix.index.name = "true\\predicted"
    matrix.to_csv(out_dir / f"{name}_confusion_matrix.csv")

    return {
        "n_samples": len(df),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=LABELS, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(truth, probabilities, labels=class_order)),
        "per_class": classification_report(
            truth, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "benchmark_1200_v1"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "lightgbm_baseline"
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {name: load_split(data_dir, name) for name in ("train", "val", "test")}
    train, val, test = (splits[name] for name in ("train", "val", "test"))
    if set(train["label"]) != set(LABELS):
        raise ValueError("Train split must contain all seven labels CLEAN and F1-F6")
    if set(val["label"]) != set(LABELS) or set(test["label"]) != set(LABELS):
        raise ValueError("Validation and test splits must contain all seven labels")

    model_params = {
        "objective": "multiclass",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 6,
        "min_child_samples": 5,
        "random_state": SEED,
        "n_jobs": 1,
        "verbosity": -1,
        "importance_type": "gain",
        "deterministic": True,
        "force_col_wise": True,
    }
    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        train[list(FEATURES)],
        train["label"],
        eval_set=[(val[list(FEATURES)], val["label"])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    model.booster_.save_model(str(out_dir / "model.txt"), num_iteration=model.best_iteration_)

    scores = {}
    for split_name, df in splits.items():
        scores[split_name] = score_split(split_name, df, model, out_dir)

    pd.DataFrame(
        {"feature": FEATURES, "gain_importance": model.feature_importances_}
    ).sort_values("gain_importance", ascending=False).to_csv(
        out_dir / "feature_importance.csv", index=False
    )

    dvc_file = ROOT / "data" / f"{data_dir.name}.dvc"
    manifest = {
        "git_commit": git_command("rev-parse", "HEAD"),
        "git_worktree_changes": bool(git_command("status", "--porcelain")),
        "training_script_sha256": sha256(Path(__file__).resolve()),
        "data_directory": str(data_dir),
        "dataset_dvc_file": str(dvc_file.relative_to(ROOT)) if dvc_file.exists() else None,
        "dataset_dvc_metadata_sha256": sha256(dvc_file) if dvc_file.exists() else None,
        "csv_sha256": {s: sha256(data_dir / f"{s}.csv") for s in splits},
        "seed": SEED,
        "features": FEATURES,
        "label_order": LABELS,
        "split_sizes": {s: len(df) for s, df in splits.items()},
        "class_counts": {s: df["label"].value_counts().reindex(LABELS, fill_value=0).to_dict() for s, df in splits.items()},
        "parameters": model_params,
        "best_iteration": int(model.best_iteration_),
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "unique_train_feature_vectors": len(feature_fingerprints(train)),
        "val_feature_overlap": overlap_details(train, val),
        "test_feature_overlap": overlap_details(train, test),
        "notes": [
            "Baseline uses 16 evidence features; target label is never included in X.",
            "Training/validation/test split is predetermined by the source benchmark.",
            "Validation used only for early stopping; test used only for evaluation.",
            "Synthetic failure signatures and repeated feature vectors may inflate measured performance.",
            "Smoke-test metrics are not representative generalization estimates.",
        ],
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(scores, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Data: {data_dir}")
    print(f"Classes: {', '.join(model.classes_)} | Features: {len(FEATURES)}")
    print(f"Best iteration (validation early stopping): {model.best_iteration_}")
    for name in ("train", "val", "test"):
        print(
            f"{name.upper():5s} n={scores[name]['n_samples']:3d} "
            f"accuracy={scores[name]['accuracy']:.4f} "
            f"macro_f1={scores[name]['macro_f1']:.4f}"
        )
    print(f"Validation overlap: {manifest['val_feature_overlap']}")
    print(f"Test overlap: {manifest['test_feature_overlap']}")
    print(f"Saved model and results: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
