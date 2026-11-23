# LightGBM + Gemini (B2) starter

Copy the files into the root of the existing PipeFail repository.

## Setup

```bash
python -m pip install -r requirements-gemini.txt
python -m pip install -e .
```

Set `GEMINI_API_KEY` securely and `GEMINI_MODEL` to an exact model ID
available to your Google AI Studio account. Check available models using `python scripts/list_gemini_models.py`. Never commit secrets.

## Run a demo

Use the existing trained LightGBM `model.txt` and `run_manifest.json`.
Full-benchmark data requires your team's DVC access.

```bash
python scripts/demo_gemini_repair.py   --model-dir results/walkthrough_v1/lightgbm   --data-dir data/benchmark_1200_v1 --split val --row 0
```

Optional `--evidence-json incident.json` supplies trusted, sanitized schema
evidence. Without this, Gemini only receives 16 numeric features, and F2
type conversion should defer for manual review.

## Compare B1 vs B2

```bash
python scripts/evaluate_gemini_repairs.py   --model-dir results/walkthrough_v1/lightgbm   --data-dir data/benchmark_1200_v1 --split val --limit 10
```

The workflow can run a sandbox-only F2 trial if called with
`candidate_parquet`, `clean_reference_parquet`, and suitable trusted
schema evidence. The CLI demos intentionally do not execute repairs.

Gemini may disagree with LightGBM's failure type. Its suggested action
is advisory; the existing runbook and sandbox checks remain authoritative.
API errors block the workflow. Human review remains required.
Do not send PII, credentials, or unsanitized logs to Gemini.
This is a starter implementation, not production deployment.

## Migration from Claude
The old Claude files can remain in the repository; the Gemini scripts do not import them.
Use a new output directory for each repeated run to preserve append-only audits.
A free-tier quota or API failure is recorded as a blocked incident, not a successful diagnosis.
