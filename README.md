# Data Pipeline Failure Agent

CS6170 - AI Capstone
Northeastern University

## Overview

PipeFail is an AI-powered system for detecting data pipeline
failures, diagnosing root causes, and recommending safe repairs.

The system analyzes pipeline evidence, identifies failures,
recommends corrective actions, and validates supported repairs
in an isolated sandbox before human approval.

## Team

- Sujith Peddireddy
- Dhruv Dolas
- Pratyush Tyagi

### Supported Failure Types

| Code | Failure |
|---|---|
| F1 | Schema change or missing column |
| F2 | Data type change |
| F3 | Unexpected null-value spike |
| F4 | Row-count loss |
| F5 | Invalid data values |
| F6 | Missing or late data arrival |

## Reproduce the Baseline

This repository includes a reproducible baseline smoke test.

It covers:
- Automated unit tests
- B0 rule-based failure detection and diagnosis
- LightGBM diagnosis model training and evaluation
- Sandbox repair validation and human-review safeguards

The smoke test uses a small dataset committed to Git.

It does not require:
- The full NYC TLC dataset
- DVC or Google Drive authentication
- Docker or Airflow
- A GPU
- Claude API credentials

### Prerequisites

- Git
- Python 3.13
- Internet access for cloning and installing dependencies
- Access to this GitHub repository

The baseline was verified using Python 3.13.14 on Windows.

### Step 1: Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/sujithpeddireddy-neu/data-pipeline-failure-agent.git
cd data-pipeline-failure-agent
```

### Step 2: Create a Virtual Environment

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Verify the interpreter:

```powershell
python --version
```

### Step 3: Install Dependencies

Install the pinned baseline dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check for dependency conflicts:

```powershell
python -m pip check
```

The requirements file contains the dependency versions
used in our verified baseline environment.

### Step 4: Run the Smoke Test

```powershell
python scripts/smoke_test.py
```

This is the main command for the CS6170 baseline walkthrough.

A successful run reports:

```text
PIPEFAIL SMOKE TEST SUMMARY

unit_tests: PASS
b0_evaluation: PASS
lightgbm_baseline: PASS
b0_repair_demo: PASS

Under five minutes: True
Overall status: PASSED
```

The smoke-test runner reports its execution time and records
the Git commit, dependency versions, input SHA-256 hashes,
individual logs, and baseline results.

Results are saved to:

```text
results/smoke_runs/<run_id>/
```

The main audit file is:

```text
results/smoke_runs/<run_id>/smoke_report.json
```

Initial dependency installation is not included in the
five-minute smoke-test execution measurement.

## Dataset and Reproducibility

### Smoke-Test Dataset

Dataset version: `pipefail-smoke-test-v1`

Source benchmark: `batch_1200_v1`

Sampling seed: `6170`

The committed smoke-test files are located at:

```text
tests/fixtures/smoke_test/
```

This directory contains:

- `train.csv`
- `val.csv`
- `test.csv`
- `manifest.json`

The dataset contains 70 synthetic examples covering seven
classes: CLEAN and F1 through F6.

Each example contains 16 numerical evidence features.

The target label is excluded from the model input.

### Dataset Splits

| Split | Examples | Examples per class |
|---|---:|---:|
| Training | 42 | 6 |
| Validation | 14 | 2 |
| Test | 14 | 2 |

The smoke-test subset was created using deterministic,
balanced sampling without replacement from the source
benchmark's existing splits.

The original benchmark uses chronological assignments:
- Training: 2019-2022
- Validation: 2023
- Test: 2024 onward

The committed smoke-test manifest documents the sampling
method, seed, feature names, source hashes, and output hashes.

Larger datasets are versioned using DVC but are not required
for the smoke test.

## Baseline Architecture

### B0: Rule-Based Detection and Diagnosis

B0 applies explainable rules to structured pipeline evidence.

These rules examine:
- Schema differences
- Column data types
- Null-rate changes
- Row-count anomalies
- Invalid values
- Data-arrival failures

Thresholds are defined in:

```text
configs/b0_rules.json
```

The calibrated F3 null-spike threshold is 3.0 percentage points.

### LightGBM Diagnosis

LightGBM classifies pipeline incidents using the same
16 numerical features.

The training implementation is:

```text
scripts/train_lightgbm_baseline.py
```

The script trains a multiclass classifier and reports
accuracy, macro-F1, and feature-vector overlap across splits.

### B0 Repair Workflow

The baseline includes an initial safe repair workflow.

The demonstration shows:

1. F2 data type change is diagnosed.
2. A type conversion is proposed.
3. The repair executes in an isolated sandbox.
4. The result is validated against expected data.
5. Human approval remains pending.

For F3 null spikes, the baseline recommends manual review
rather than automatically quarantining potentially valid rows.

No live data is modified by the smoke-test demonstration.

## Baseline Results

### B0 Smoke-Test Results

| Metric | Training | Validation |
|---|---:|---:|
| Detection precision | 100.00% | 100.00% |
| Detection recall | 91.67% | 83.33% |
| Diagnosis accuracy | 92.86% | 85.71% |

### LightGBM Smoke-Test Results

| Metric | Training | Validation | Test |
|---|---:|---:|---:|
| Accuracy | 92.86% | 92.86% | 92.86% |
| Macro-F1 | 92.38% | 92.38% | 92.38% |

These results were reproduced on Git commit:

`32bd61bf648b0c615d1e8d8219c76958768dfb17`

The integrated smoke test completed in 4.96 seconds in
the development environment.

A separate earlier fresh-clone verification completed in
35.08 seconds.

These measurements are specific to those environments.

## Known Limitations

The smoke-test dataset is small and synthetic. Its results
demonstrate reproducibility, not production performance.

The source benchmark contains repeated feature vectors.

In the smoke-test subset:
- 7 of 14 validation examples have feature vectors seen in training.
- 8 of 14 test examples have feature vectors seen in training.

This limits conclusions about generalization.

The LightGBM smoke-test script also evaluates the test split.
Therefore, that split cannot be described as untouched.

The full B0 rule evaluation uses a larger 1,200-case synthetic
benchmark, but the default smoke test does not require it.

The repair workflow is an initial implementation. Successful
sandbox validation does not authorize changes to production data.

Full Airflow orchestration and Claude-assisted repair reasoning
are not demonstrated by this smoke test.

## Development Status

Implemented:
- TLC data collection and profiling
- Failure injection F1-F6
- Experiment generation and auditing
- DVC dataset versioning
- Evidence collection and feature extraction
- B0 detection and diagnosis
- LightGBM diagnosis
- Initial sandbox repair workflow
- Schema-contract validation
- Human-review safeguards
- Reproducible baseline smoke test

Planned:
- Comprehensive repair evaluation across F1-F6
- Airflow orchestration integration
- Advanced AI-assisted repair recommendations
- Comparative architecture evaluation
