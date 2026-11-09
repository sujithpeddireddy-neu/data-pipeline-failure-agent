"""
PipeFail reproducible baseline smoke test.

Runs entirely against committed smoke-test fixtures.
No TLC raw-data directory, DVC remote, Docker, GPU,
or external API credentials are required.
"""

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "smoke_test"
START = time.perf_counter()

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT = ROOT / "results" / "smoke_runs" / RUN_ID
OUTPUT.mkdir(parents=True, exist_ok=False)


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT_INSTALLED"


inputs = [
    "tests/fixtures/smoke_test/manifest.json",
    "tests/fixtures/smoke_test/train.csv",
    "tests/fixtures/smoke_test/val.csv",
    "tests/fixtures/smoke_test/test.csv",
    "configs/b0_rules.json",
    "pyproject.toml",
    "scripts/evaluate_b0.py",
    "scripts/train_lightgbm_baseline.py",
    "scripts/demo_b0_workflow.py",
]

missing = [name for name in inputs if not (ROOT / name).is_file()]

if missing:
    raise SystemExit(f"Missing required files: {missing}")

git_commit = git_output("rev-parse", "HEAD")
git_status = git_output("status", "--porcelain")

report = {
    "run_id": RUN_ID,
    "git_commit": git_commit,
    "git_worktree_clean": not bool(git_status),
    "python_version": sys.version,
    "dependencies": {
        name: package_version(name)
        for name in [
            "pytest",
            "pandas",
            "pyarrow",
            "duckdb",
            "lightgbm",
            "scikit-learn",
        ]
    },
    "input_sha256": {
        name: sha256(ROOT / name)
        for name in inputs
    },
    "steps": [],
}

print("=" * 65)
print("PIPEFAIL — REPRODUCIBLE BASELINE SMOKE TEST")
print("=" * 65)

print("Git commit:", git_commit)
print("Git worktree clean:", report["git_worktree_clean"])
print("Input files verified:", len(inputs))
print("Output:", OUTPUT)


commands = [
    (
        "unit_tests",
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ],
    ),
    (
        "b0_evaluation",
        [
            sys.executable,
            "scripts/evaluate_b0.py",
            "--data-dir",
            str(FIXTURES),
            "--config",
            "configs/b0_rules.json",
            "--splits",
            "train",
            "val",
            "--output",
            str(OUTPUT / "b0_metrics.json"),
        ],
    ),
    (
        "lightgbm_baseline",
        [
            sys.executable,
            "scripts/train_lightgbm_baseline.py",
            "--data-dir",
            str(FIXTURES),
            "--output-dir",
            str(OUTPUT / "lightgbm"),
        ],
    ),
    (
        "b0_repair_demo",
        [
            sys.executable,
            "scripts/demo_b0_workflow.py",
        ],
    ),
]


success = True

for name, command in commands:

    print(f"\n{'=' * 65}")
    print("RUNNING:", name)
    print("=" * 65)

    step_start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        returncode = completed.returncode
        output_text = completed.stdout + completed.stderr

    except subprocess.TimeoutExpired:
        returncode = -1
        output_text = "ERROR: Command exceeded 300 seconds."

    duration = round(time.perf_counter() - step_start, 3)

    log_path = OUTPUT / f"{name}.log"
    log_path.write_text(output_text, encoding="utf-8")

    print(output_text)
    print(f"Duration: {duration:.2f} seconds")

    step_passed = returncode == 0

    if name == "b0_repair_demo":
        if "Live data modified: False" not in output_text:
            step_passed = False

        if "F2 validation: PASSED" not in output_text:
            step_passed = False

        if "F3 repair proposal: MANUAL_REVIEW" not in output_text:
            step_passed = False

    report["steps"].append({
        "name": name,
        "command": command,
        "returncode": returncode,
        "passed": step_passed,
        "duration_seconds": duration,
        "log_file": str(log_path.relative_to(ROOT)),
    })

    if not step_passed:
        print(f"\nFAILED: {name}")
        success = False
        break


elapsed = round(time.perf_counter() - START, 3)

report["total_duration_seconds"] = elapsed
report["under_five_minutes"] = elapsed < 300
report["all_steps_passed"] = success
report["status"] = (
    "PASSED"
    if success and elapsed < 300
    else "FAILED"
)

report_path = OUTPUT / "smoke_report.json"

report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 65)
print("PIPEFAIL SMOKE TEST SUMMARY")
print("=" * 65)

for step in report["steps"]:
    status = "PASS" if step["passed"] else "FAIL"
    print(f"{step['name']}: {status}")

print(f"\nTotal execution time: {elapsed:.2f} seconds")
print("Under five minutes:", elapsed < 300)
print("Overall status:", report["status"])
print("Audit report:", report_path)

if not report["git_worktree_clean"]:
    print(
        "\nNOTE: Git worktree contains uncommitted files. "
        "Commit the smoke-test runner before claiming "
        "a clean-commit reproduction."
    )

if report["status"] != "PASSED":
    sys.exit(1)
