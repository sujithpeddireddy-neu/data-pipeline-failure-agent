# PipeFail — Baseline and Learned-Model Code Walkthrough

**CS6170 AI Capstone | November 2026 | Baseline and learned-model walkthrough**

**Team:** Dhruv Dolas, Sujith Peddireddy, Pratyush Tyagi  
**Git repository:** `sujithpeddireddy-neu/data-pipeline-failure-agent`  
**Baseline training/evaluation code revision:** `fc47e8c41c7446fdb402b328f42e78939da8e564`  
**LightGBM repair integration code revision:** `bd6dcc2e5247c849426ee2ec799888a39e719174` on `feature/lightgbm-repair`.  
**Test-set status:** previously inspected; subsequent full-benchmark runs are **reproducibility checks**, not unbiased first-use tests.

## What is implemented

PipeFail accepts 16 numeric features derived from pipeline evidence. It provides two comparable diagnosis paths:

| Component | Implemented function | Result artifact |
|---|---|---|
| **B0 rulebook** | Deterministic detection and F1–F6/CLEAN diagnosis using fixed thresholds | `docs/experiments/walkthrough_v1/b0/b0_metrics.json` |
| **LightGBM learned baseline** | Supervised seven-class diagnosis (CLEAN and F1–F6); `predicted_label != CLEAN` is the **derived binary detection decision** | `docs/experiments/walkthrough_v1/lightgbm/metrics.json`, `detection_summary.json` |
| **Constrained repair runbook** | Converts LightGBM's predicted class to a proposed, allowlisted action, manual review, or no action | `docs/experiments/walkthrough_v1/lightgbm/test_repair_suggestions.csv` and `test_repair_summary.json` |
| **Sandbox and human gate** | F2 trusted-evidence type conversion, isolated execution, validation; human review stays pending | `docs/experiments/walkthrough_v1/lightgbm/f2_sandbox_audit.json` |

LightGBM **does not generate arbitrary repair code**. Detection is derived from its seven-class output, and the repair suggestion comes from a separate rulebook. F2 sandbox evidence is one demonstration, **not** a 400-case repair success test. The full pipeline's Airflow/production deployment and autonomous repair are not established by these results.

## Cold-clone demo (no credentials or GPU)

From a clean clone on Windows with Python 3.13:

```powershell
git clone https://github.com/sujithpeddireddy-neu/data-pipeline-failure-agent.git
cd data-pipeline-failure-agent
git switch feature/lightgbm-repair
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test runs committed 70-example fixtures, B0 evaluation, LightGBM training/evaluation, and the B0 repair demonstration; it does not automatically run the full-benchmark LightGBM repair recommendation demo. A Windows reference run completed in **33.62 seconds**; execution time depends on the machine and environment.

## Full benchmark reproduction (requires authorized DVC access)

```powershell
python -m pip install 'dvc[gdrive]' 
dvc pull data/benchmark_1200_v1.dvc
python scripts/evaluate_b0.py --data-dir data/benchmark_1200_v1 --config configs/b0_rules.json --splits train val test --output results/walkthrough_v1/b0_metrics.json
python scripts/train_lightgbm_baseline.py --data-dir data/benchmark_1200_v1 --output-dir results/walkthrough_v1/lightgbm
python scripts/evaluate_lightgbm_repairs.py --data-dir data/benchmark_1200_v1 --split test --model-dir results/walkthrough_v1/lightgbm --output-dir results/walkthrough_v1/lightgbm/repair_recommendations
python scripts/demo_lightgbm_repair.py --data-dir data/benchmark_1200_v1 --model-dir results/walkthrough_v1/lightgbm
```

**Dataset split:** 600 train (2019–2022), 200 validation (2023), 400 test (2024–Sep 2026), totaling 1,200 synthetically injected incidents derived from real TLC Parquet samples. An audit of all planned cases reported **zero time-split violations**. The model is trained on incident features, not individual raw TLC taxi trips.

## Recorded numbers (full-benchmark reproducibility runs)

| Test measure (400 cases) | B0 rulebook | LightGBM learned baseline |
|---|---:|---:|
| Failure detection precision | 279/279 (100%) | 300/300 (100%) |
| Failure detection recall | 279/300 (93%) | 300/300 (100%) |
| Diagnosis accuracy | **379/400 (94.75%)** | **400/400 (100%)** |
| F4 recall | 40/50 (80%) | 50/50 (100%) |
| F5 recall | 39/50 (78%) | 50/50 (100%) |
| Test macro-F1 | Not recorded as the same metric | 1.0000 |

**LightGBM repair proposal results:** 400 test incidents, 300 `REVIEW_REQUIRED`, 100 `NO_ACTION`; 250 `MANUAL_REVIEW` runbook actions (including 100 CLEAN), 50 `RERUN_MONTH`, 100 `PAUSE_DOWNSTREAM`. These are **proposals**, not performed repairs. A separate F2 conversion passed sandbox validation with `PENDING_HUMAN_REVIEW`, and no live data modification. **71/71** unit tests passed on the developer's Windows environment with Python 3.13.14.

## Evaluation results and artifacts

The following results link each reported metric to the corresponding experiment output, model artifact, predictions, or validation record.

| Evidence | Inspect on GitHub | What it shows |
|---|---|---|
| B0 evaluation | [B0 metrics JSON](docs/experiments/walkthrough_v1/b0/b0_metrics.json) | Split-level detection and diagnosis metrics and F4/F5 misses |
| Learned diagnosis evaluation | [LightGBM metrics JSON](docs/experiments/walkthrough_v1/lightgbm/metrics.json) | Train/validation/test accuracy and related evaluation metrics |
| Learned detection | [Detection summary JSON](docs/experiments/walkthrough_v1/lightgbm/detection_summary.json) | CLEAN vs. F1–F6 binary detection: TP 300, FP 0, FN 0, TN 100 |
| Reproducibility | [Training manifest JSON](docs/experiments/walkthrough_v1/lightgbm/run_manifest.json) · [Dataset provenance JSON](docs/experiments/walkthrough_v1/provenance.json) | Seed, configuration, hashes, model revision, feature overlap, and export revision |
| Model artifact | [Trained LightGBM model](docs/experiments/walkthrough_v1/lightgbm/model.txt) · [Feature importance CSV](docs/experiments/walkthrough_v1/lightgbm/feature_importance.csv) | Learned model and feature rankings |
| Predictions | [Test predictions CSV](docs/experiments/walkthrough_v1/lightgbm/test_predictions.csv) · [Test confusion matrix CSV](docs/experiments/walkthrough_v1/lightgbm/test_confusion_matrix.csv) | Per-incident predictions and seven-class error counts |
| Repair proposals | [Per-incident repair suggestions CSV](docs/experiments/walkthrough_v1/lightgbm/test_repair_suggestions.csv) · [Repair summary JSON](docs/experiments/walkthrough_v1/lightgbm/test_repair_summary.json) | Runbook actions and review statuses across 400 cases; **no bulk repair execution** |
| Sandbox safety | [F2 sandbox audit JSON](docs/experiments/walkthrough_v1/lightgbm/f2_sandbox_audit.json) | One isolated `CONVERT_TYPE` trial, validation pass, and human review still pending |

**Reproducibility.** Versioned DVC pointers `data/benchmark_1200_v1.dvc` and `data/raw/yellow_taxi.dvc`, together with `.dvc/config`, identify the dataset objects independently from the Git-tracked code and small result files. `scripts/train_lightgbm_baseline.py` writes a manifest with configuration, seed, model/source revision and input hashes; `scripts/evaluate_lightgbm_repairs.py` checks the benchmark hash against the model's training manifest. This makes the result files traceable to a model and data version without committing large source datasets to Git.


## Verification and safeguards

- **Unit tests:** 71/71 passed on Windows, Python 3.13.14. The F2 LightGBM sandbox demonstration passed validation and left `PENDING_HUMAN_REVIEW` with `live_data_modified: false`.
- **Cold run:** `python scripts/smoke_test.py` uses the Git-committed 70-case fixture without GPU or DVC credentials. A prior developer run took approximately 34 seconds. The README gives the pinned installation path; the cold-clone commands above let the instructor reproduce the smoke test on a separate machine.
- **Data integrity:** The full-benchmark evaluation used the 1,200-case dataset identified by SHA-256 hashes and tracked through DVC. The test feature-vector overlap and repeated test usage are disclosed above.
- **Scope:** The published test results do not establish real-world failure-detection accuracy, bulk repair success rate, autonomous deployment, or production safety. Sandbox validation does not bypass human approval.
- **Access:** The Git repository and committed smoke-test fixture are sufficient for the credential-free demo. Reproducing the full benchmark from DVC additionally requires permission to access the configured data remote.

## Post-meeting implementation updates

Following the review meeting, the team completed two connected components and recorded their outputs in the files above:

1. **LightGBM learned baseline for detection and diagnosis.** Trained a seven-class model on the 16-feature benchmark to distinguish CLEAN from F1–F6. Non-CLEAN predictions act as learned failure detection; the predicted F1–F6 label provides diagnosis. The [LightGBM metrics](docs/experiments/walkthrough_v1/lightgbm/metrics.json), [detection summary](docs/experiments/walkthrough_v1/lightgbm/detection_summary.json), and [test predictions](docs/experiments/walkthrough_v1/lightgbm/test_predictions.csv) show the evaluated outputs. Model training/evaluation provenance is recorded at `fc47e8c`.
2. **Repair recommendations and gated sandbox validation.** Integrated LightGBM diagnosis with the existing constrained runbook, produced [400 repair/no-action proposals](docs/experiments/walkthrough_v1/lightgbm/test_repair_summary.json), and validated one F2 type-conversion trial in an isolated sandbox. The [sandbox audit](docs/experiments/walkthrough_v1/lightgbm/f2_sandbox_audit.json) records that approval remains pending and no live data was modified. The integration code was committed at `bd6dcc2`; 71 unit tests passed afterward.

