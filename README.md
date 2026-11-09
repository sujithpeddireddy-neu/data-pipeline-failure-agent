# PipeFail — Data Pipeline Failure Agent

**CS6170 AI Capstone · Northeastern University**

PipeFail is a data-pipeline failure detection, diagnosis, and repair-recommendation system. It converts pipeline evidence into structured incident features, identifies supported failure types, and routes proposed repairs through validation and human review. The current implementation provides a deterministic rule-based baseline (B0), a LightGBM diagnosis comparison, and an isolated repair demonstration. Full production orchestration and foundation-model-assisted repair are not part of the reproducible baseline.

**Team:** Sujith Peddireddy · Dhruv Dolas · Pratyush Tyagi

## Supported incidents

| Label | Condition |
|---|---|
| CLEAN | No injected failure |
| F1 | Missing or renamed expected column (schema change) |
| F2 | Unexpected column-type change |
| F3 | Increase in missing-value rate |
| F4 | Unexpected row loss |
| F5 | Invalid data values |
| F6 | Missing or late data arrival |

## Quick start: reproduce the baseline

The baseline smoke test uses **70 labeled examples committed directly to Git** under `tests/fixtures/smoke_test/`. **No DVC download, Google Drive access, Docker, GPU, or external AI API key is required.** The complete smoke test is started with a single command after installation.

### 1. Clone the repository

```bash
git clone https://github.com/sujithpeddireddy-neu/data-pipeline-failure-agent.git
cd data-pipeline-failure-agent
git rev-parse HEAD
```

The final command prints the exact checked-out source revision. Read access to the repository is required if it is private.

### 2. Set up Python

Python **3.13.14** was used for the verified Windows baseline. The package declares Python `>=3.11`, but other versions and platforms have not been verified to the same extent.

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

If PowerShell blocks script activation, a session-only workaround is `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, followed by the activation command. Alternatively, call `.\.venv\Scripts\python.exe` explicitly without activating.

**macOS / Linux (bash or zsh)**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

### 3. Install baseline dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

`pyproject.toml` defines the installable package; `requirements.txt` is a pinned dependency snapshot from the verified baseline environment. DVC is **not** required for this step and is installed separately for full-dataset access.

### 4. Execute the smoke test

```bash
python scripts/smoke_test.py
```

The runner executes the unit tests, B0 rules evaluation, LightGBM training/evaluation, and the B0 sandbox-repair demonstration. A successful execution ends with:

```text
PIPEFAIL SMOKE TEST SUMMARY
unit_tests: PASS
b0_evaluation: PASS
lightgbm_baseline: PASS
b0_repair_demo: PASS
Under five minutes: True
Overall status: PASSED
```

Reports and logs are generated at `results/smoke_runs/<UTC_run_id>/`, including `smoke_report.json`, `b0_metrics.json`, and the LightGBM model/results directory. The report records the Git commit, worktree status, Python and selected package versions, SHA-256 hashes of nine inputs, execution commands, and durations. The **five-minute threshold applies to smoke-test execution**, not repository cloning or dependency installation.

The smoke test can be repeated to compare metrics and recorded hashes across runs. Output directories and elapsed times can differ.

## Full dataset: DVC versioning and reproduction (optional)

This section restores the **versioned research data** for benchmark evaluation or further experiments. It is **not a prerequisite for the smoke test**. The DVC remote is a team-managed **private Google Drive folder**; access must be granted separately.

### DVC-tracked datasets

The repository commits DVC pointer files; large data objects are stored remotely and restored by `dvc pull`.

| Pointer file | Restored location | Contents |
|---|---|---|
| `data/raw/yellow_taxi.dvc` | `data/raw/yellow_taxi/` | Historical NYC TLC Yellow Taxi Parquet files (~5.45 GB across the tracked directory) |
| `data/benchmark_1200_v1.dvc` | `data/benchmark_1200_v1/` | 1,200-case synthetic benchmark CSVs, index, and metadata |
| `data/experiment_artifacts_1200_v1.zip.dvc` | `data/experiment_artifacts_1200_v1.zip` | Archived experiment artifacts (~68 MB) |
| `data/profiles.dvc` | `data/profiles/` | Dataset profiling outputs |
| `data/smoke_test.dvc` | `data/smoke_test/` | Separately DVC-tracked smoke-test data; the baseline uses the **Git-committed** fixture instead |

The remote definition is committed in `.dvc/config` under the name **`gdrive`**. The `.dvc` files specify object hashes and sizes. Checking out a particular Git revision selects the corresponding DVC pointers; `dvc pull` downloads those referenced objects into the local DVC cache and materializes them in the workspace. Git and DVC therefore version **different parts** of the project.

### 1. Install DVC with Google Drive support

Execute from the repository root with the Python virtual environment active:

```bash
python -m pip install "dvc[gdrive]"
dvc version
dvc remote list
```

The baseline's pinned `requirements.txt` does not include DVC or its Google Drive plugin. Installing DVC may add or update dependencies; for a strictly unchanged baseline environment, use a separate virtual environment for full-data operations.

The remote must already appear as `gdrive`; **do not run `dvc init` or create a second remote** when using this repository.

### 2. Obtain access to the remote

Access requires **both** of the following:

1. **Google Drive folder permission.** The data owner must share the folder referenced by `.dvc/config` with the Google account used for authorization. Viewer access is sufficient for downloading.
2. **OAuth authorization.** The Google account must be authorized to use the configured OAuth app (including inclusion in **Test users** if the custom app remains in Testing mode).

Google may block DVC's default OAuth application. In that case, the team administrator must create a Google Cloud OAuth **Desktop** client with the Google Drive API enabled and supply the client ID and secret through a secure channel. Each collaborator configures that client **locally**:

```bash
dvc remote modify --local gdrive gdrive_client_id "<GOOGLE_OAUTH_CLIENT_ID>"
dvc remote modify --local gdrive gdrive_client_secret "<GOOGLE_OAUTH_CLIENT_SECRET>"
```

The `--local` flag writes `.dvc/config.local`, which is Git-ignored. Never commit OAuth secrets, browser authorization codes, or cached user tokens. Each collaborator signs in with their **own Google account** when the authorization page opens.

For details, refer to the [DVC Google Drive remote documentation](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive).

### 3. Restore the entire versioned dataset

```bash
dvc pull
```

This downloads **all data referenced by the current checkout's DVC pointers**. Ensure that sufficient disk space and network bandwidth are available for the multi-gigabyte raw dataset plus the other artifacts and local DVC cache.

To restore only a particular component, use its pointer file as a target:

```bash
# Full 1,200-case benchmark without raw monthly TLC files
dvc pull data/benchmark_1200_v1.dvc

# Historical raw TLC Parquet files only
dvc pull data/raw/yellow_taxi.dvc
```

Verify the working data state:

```bash
dvc status
```

`dvc pull` restores the contents identified by the current pointer hashes; it **does not regenerate synthetic experiments from the raw records**. This is the preferred route to reproduce the exact versioned inputs for evaluation.

### 4. Evaluate the restored 1,200-case benchmark

Once `data/benchmark_1200_v1/` has been restored, the baseline can run on the larger dataset:

```bash
# Rule-based B0: training and validation splits
python scripts/evaluate_b0.py \
  --data-dir data/benchmark_1200_v1 \
  --config configs/b0_rules.json \
  --splits train val \
  --output results/b0_rules/full_benchmark_metrics.json

# LightGBM diagnosis comparison (trains on train, early-stops on val)
python scripts/train_lightgbm_baseline.py \
  --data-dir data/benchmark_1200_v1 \
  --output-dir results/lightgbm_full_benchmark
```

These examples use **bash/zsh line continuation** (`\`). In Windows PowerShell, place each command on one line or use PowerShell backticks for line continuation.

**Evaluation caveat:** The LightGBM script also reports test-set performance. Its smoke/full-benchmark test results should not be described as never-inspected locked-test results. Synthetic feature signatures can repeat across splits.

### 5. Rebuild synthetic experiments from the raw TLC version (advanced)

Restoring a DVC snapshot and **regenerating** that snapshot are separate procedures. Regeneration requires the 93 monthly TLC Parquet files used by the benchmark (January 2019–September 2026) in `data/raw/yellow_taxi/` and can require substantially more time, disk space, and compute than the smoke test.

The experiment generator requires an explicit data directory. Generate a new, non-overwriting batch:

```bash
# Inspect the 1,200-case plan without generating Parquet artifacts
python scripts/generate_experiment_dataset.py \
  --data-dir data \
  --batch-name reproduced_1200_v1 \
  --plan-seed 6170 \
  --sample-size 1000 \
  --dry-run

# Generate the batch (resource-intensive)
python scripts/generate_experiment_dataset.py \
  --data-dir data \
  --batch-name reproduced_1200_v1 \
  --plan-seed 6170 \
  --sample-size 1000

# Validate the generated plan, splits, labels, and features
python scripts/audit_experiment_dataset.py \
  --data-dir data \
  --batch-name reproduced_1200_v1
```

Output is written to `data/experiments/reproduced_1200_v1/`. The generator enforces chronological source-month assignments: **train = 2019–2022**, **validation = 2023**, and **test = January 2024–September 2026**. The plan seed controls experiment selection and injection parameters. Regeneration was **not** part of the five-minute cold-run verification; exact byte-for-byte equality with the stored snapshot is not guaranteed across environments.

### Troubleshooting DVC access

| Symptom | Check |
|---|---|
| Google displays **“Action blocked”** | Use the team-managed custom OAuth Desktop client, with the Google Drive API enabled and the user permitted by the OAuth consent configuration. |
| **Authentication successful**, followed by Drive **404 / File not found** | Verify that the authenticated account has direct access to the folder referenced in `.dvc/config`, that the folder still exists, and that the correct Google account was selected. OAuth test-user access alone does not grant Drive folder access. |
| `dvc: command not found` | Install `dvc[gdrive]` in the active environment and verify `dvc version`. |
| `dvc pull` reports missing objects | Confirm that the current DVC pointers have been uploaded to the remote; successful OAuth alone does not guarantee that every referenced object exists. |

## Dataset, splits, and leakage notes

The main benchmark contains **1,200 synthetic cases**: **600 training**, **200 validation**, and **400 test** cases. It includes 300 CLEAN examples and 150 examples for each failure type (F1–F6). The smoke-test subset uses label-balanced sampling with seed **6170**: 42 train, 14 validation, and 14 test examples (6/2/2 per class). `scripts/create_smoke_test.py` implements this selection and preserves the source benchmark's split assignments.

The smoke fixture's `manifest.json` records the source dataset, sampling method, feature names, and SHA-256 hashes of the source and output CSVs. Each model example has **16 numerical evidence features**; the ground-truth label is not an input feature.

**Leakage limitations:** The benchmark is synthetic and contains repeated feature vectors. In the smoke subset, 7 of 14 validation rows and 8 of 14 test rows have feature vectors that also appear in training. Chronological separation does not eliminate synthetic-template overlap. These metrics demonstrate executable behavior, not production generalization.

**Data provenance:** The underlying trip records are published by the [NYC Taxi and Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Use of the source records is subject to the applicable [City of New York terms](https://www.nyc.gov/home/terms-of-use.page). The original source is not included in Git; the DVC remote has separate access controls.

## Baseline implementation

**B0 detection and diagnosis.** `src/pipefail/detection/rules.py` applies deterministic rules configured in `configs/b0_rules.json`; `src/pipefail/diagnosis/runbook.py` assigns supported diagnoses or defers ambiguous cases. For F3, the configured null-rate spike threshold is **3.0 percentage points**. Added columns alone are not automatically treated as failures.

**LightGBM diagnosis.** `scripts/train_lightgbm_baseline.py` loads 16 structured features and trains a multiclass classifier for CLEAN and F1–F6, using validation-based early stopping. It records model configuration, seed, dataset hashes, and feature-vector overlap.

**Sandbox repair and review.** `scripts/demo_b0_workflow.py` demonstrates an F2 type-conversion proposal validated on isolated Parquet data and an F3 manual-review path. `src/pipefail/repair/` contains proposal execution and validation logic; `src/pipefail/review/` records human decisions. A sandbox pass does **not** authorize deployment to live data.

## Recorded smoke-test results

The following results are from the integrated baseline on Git revision `32bd61bf648b0c615d1e8d8219c76958768dfb17`, before later documentation-only commits. A smoke run on that development environment finished in **4.96 seconds**; a prior fresh-clone run of the feature branch finished in **35.08 seconds**. Reproduction times may differ by machine.

| B0 metric | Train (42) | Validation (14) |
|---|---:|---:|
| Detection precision | 100.00% | 100.00% |
| Detection recall | 91.67% | 83.33% |
| Diagnosis accuracy | 92.86% | 85.71% |

| LightGBM metric | Train (42) | Validation (14) | Test (14) |
|---|---:|---:|---:|
| Accuracy | 92.86% | 92.86% | 92.86% |
| Macro-F1 | 92.38% | 92.38% | 92.38% |

The reported values are **smoke-test checks on a small synthetic dataset**, not validated operational performance estimates.

## Development status and boundaries

| Implemented | Not yet demonstrated end-to-end |
|---|---|
| Controlled failure injection (F1–F6) and feature extraction | Production Airflow orchestration |
| B0 detection, runbook diagnosis, and evaluation | Additional learned detection architectures |
| LightGBM diagnosis comparison | Claude-assisted repair reasoning |
| Initial F2 sandbox validation and F3 manual-review path | Comprehensive repair-success evaluation across all failure types |
| DVC tracking of research data and credential-free smoke test | Autonomous repair deployment (intentionally outside the current demo) |

The smoke test demonstrates a **reproducible baseline workflow**, not a complete production data-pipeline agent.
