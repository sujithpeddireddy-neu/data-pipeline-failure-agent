# PipeFail — Release Plan (v0)

**Course:** CS6170 AI Capstone, Northeastern University  
**Project:** Data Pipeline Failure Agent (PipeFail)  
**Team:** Dhruv Dolas, Sujith Peddireddy, Pratyush Tyagi  
**Date:** November 24, 2026  
**Repository:** `sujithpeddireddy-neu/data-pipeline-failure-agent`  
**Integration branch:** `feature/week11-results`  
**Reference commit at the local B0 fallback verification:** `f73a49c3c2f79ae779cef1bcf0fca84c47587ea7` (release-plan commit on `feature/week11-results`; later documentation revisions will have different commit IDs)

## Release Decision and Scope

**Decision: Conditional approval for an offline, human-supervised capstone demonstration; no authorization for autonomous production repair.**

PipeFail supports detection, diagnosis, and constrained repair recommendations for data pipeline incidents. Its intended operators are on-call data engineers and pipeline maintainers. Diagnosis and repair outputs remain advisory. Any proposed repair requires sandbox validation and an explicit human decision before execution.

The B0 rulebook serves as the reference baseline and rollback target. B1 (LightGBM with constrained runbook) is an evaluated candidate for supervised use. B2 (LightGBM with Gemini-assisted diagnosis/repair workflow) remains experimental because its API reliability, evaluation coverage, and end-to-end repair safety are not yet sufficient for promotion.

Production deployment, unrestricted SQL or shell execution, deletion actions, autonomous modification of live data, and private enterprise incidents are outside the approved project scope.

## 1. Serving Pattern and Version Control

| Architecture | Serving pattern | Implementation | Status |
|---|---|---|---|
| **B0 — Rulebook** | **Embedded** in the Python application | Deterministic classification of `CLEAN` and F1–F6 from 16 numerical evidence features; constrained runbook actions. Implementation: `src/pipefail/detection/rules.py`, `src/pipefail/diagnosis/runbook.py`; configuration: `configs/b0_rules.json`. | Established baseline and rollback target. |
| **B1 — LightGBM + Runbook** | **Embedded** model inference and application-level runbook | Seven-class LightGBM prediction followed by allowlisted repair proposals, manual review, or no action. A separate F2 type-conversion example passed sandbox validation with approval pending. | Offline-evaluated candidate; production repair disabled. |
| **B2 — LightGBM + Gemini** | **Separate hosted API service**, orchestrated by local Python components | Gemini integration: `src/pipefail/diagnosis/gemini_agent.py` and `src/pipefail/workflows/gemini_workflow.py`. The evaluation records the requested model identifier `gemini-3.5-flash`. | Preliminary evaluation; not eligible for release. |

The recorded Gemini identifier represents the **requested model**, not a verified immutable provider-side version pin. The exact supported API version and prompt revision must be verified and retained before B2 promotion. The preliminary results use prompt revision `before_failure_class_definitions`; the subsequently updated prompt has not yet completed verification.

Release artifacts must identify the Git commit, dataset version and hash, feature schema, preprocessing configuration, B0 rules, LightGBM model and thresholds, and—where applicable—the Gemini model identifier and prompt revision. Model-only rollback is insufficient when the associated preprocessing or configuration changes.

Deployment and release are distinct: executing a candidate in an isolated environment does not authorize operators to act on its output.

## 2. Staged Rollout

The proposed strategy combines offline evaluation, **shadow-style batch replay**, and supervised demonstration. Because the project processes batch incidents without live production users, the replay stage is not equivalent to shadowing real production traffic. Live canary testing, A/B testing, and general production release are not planned.

| Stage | Audience and output visibility | Duration and expected cases | Promotion requirement | Status |
|---|---|---|---|---|
| **0 — Local verification** | Development team; outputs remain local | One verification session using the **70-case** committed smoke-test dataset and unit tests | All tests pass; output schema and safety controls remain valid | **71/71 unit tests passed**; smoke-test workflow documented. |
| **1 — Offline comparison** | Project team reviewing stored results | One evaluation cycle: **400** benchmark test incidents for B0/B1; **14** validation incidents for preliminary B2 analysis | Document overall and per-failure-type metrics, errors, provenance, and limitations | B0/B1 evaluated; B2 full benchmark outstanding. |
| **2 — Shadow-style batch replay** | Evaluators inspect logs; no candidate recommendation reaches an operational decision-maker | Proposed isolated batch cycle of **at least 400 fresh labeled incidents**, including 100 `CLEAN` and 50 for each F1–F6 class | Section 3 gates met; latency, deferrals, disagreements, and safety incidents reviewed | Planned; not yet executed. |
| **3 — Limited supervised demonstration** | Team and instructor acting as reviewing engineers; candidate outputs explicitly marked experimental | Proposed single demonstration session with approximately **20 synthetic incidents** spanning `CLEAN` and F1–F6 | Sandbox evidence, human decision, and audit trail demonstrated for proposed actions | Planned; selected F2 sandbox checks already demonstrated. |
| **4 — Production release** | Operational engineers or downstream data consumers | **Not scheduled; zero authorized production cases** | Separate authorization, fresh evaluation, privacy review, monitoring, and verified rollback | Outside current scope. |

For a future batch preprocessing update, **blue/green-style switching** is preferred: the revised job runs in an isolated environment, its output is validated, and downstream consumers are switched only after approval. The previous validated batch output must remain available for restoration. This control is proposed rather than implemented as a production switching mechanism.

## 3. Promotion Gates and Evaluation Evidence

Promotion assessments must compare candidates on the **same evaluation cases** and report counts by `CLEAN`, F1, F2, F3, F4, F5, and F6. These categories represent technical incident types, not demographic groups. The planned minimum is **400 labeled incidents**: 100 `CLEAN` cases and 50 cases for each failure type. New source periods and previously unseen feature combinations should be included where feasible.

### 3.1 Full Benchmark: B0 and B1

The synthetic benchmark contains **1,200 incidents** derived from NYC TLC Yellow Taxi records: 600 training, 200 validation, and 400 testing cases. The chronological test split has been reused for reproducibility checks and therefore does not constitute a fresh, untouched holdout.

| Measure | B0 Rulebook | B1 LightGBM |
|---|---:|---:|
| Test diagnosis accuracy (400 incidents) | **379/400 (94.75%)** | **400/400 (100%)** |
| Failure detection precision | 279/279 (100%) | 300/300 (100%) |
| Failure detection recall | 279/300 (93%) | 300/300 (100%) |
| F4 recall | 40/50 (80%) | 50/50 (100%) |
| F5 recall | 39/50 (78%) | 50/50 (100%) |

The results demonstrate strong performance on the available synthetic benchmark, but do not establish production reliability. Multiple validation and test examples share identical complete feature vectors with training examples, and the reserved test set has been inspected previously.

### 3.2 Smoke-Test Comparison: B0, B1, and B2

The independent **70-case smoke-test fixture** comprises 42 training, 14 validation, and 14 testing incidents. The following results relate to its **14-case validation subset**, not to the 400-case benchmark.

| Measure | B0 Rulebook | B1 LightGBM | B2 LightGBM + Gemini |
|---|---:|---:|---:|
| Validation diagnosis accuracy | **12/14 (85.71%)** | **13/14 (92.86%)** | **7/14 (50.00%)** |
| Validation diagnosis coverage | 100% | 100% | **8/14 (57.14%)** |
| Accuracy on non-deferred outputs | 85.71% | 92.86% | **7/8 (87.50%)** |
| Deferrals | 0 | 0 | **6/14 (42.86%)** |
| API failure rate | N/A | N/A | **6/14 (42.86%)** |

B1 also achieved **13/14 (92.86%)** on the smoke-test testing subset. A corresponding full test evaluation for B2 has not been completed. Comparisons across the smoke-test validation subset and the 400-case benchmark must remain separate.

### 3.3 B2 API Reliability and Safety

| Measure | Recorded result |
|---|---:|
| Validation incidents | 14 |
| Total API attempts | 28 |
| Successful diagnosis responses | 8 |
| Incidents requiring retries | 8 |
| Incidents recovered through retries | 2 |
| Retry recovery rate | 25% |
| Incidents with exhausted retries | 6 |
| Maximum attempts per incident | 3 |
| Configured request timeout | 30 seconds |
| Average diagnosis latency | **37.32 seconds** |
| Maximum diagnosis latency | **94.26 seconds** |
| Total diagnosis latency | **522.47 seconds** |
| Reported free-tier daily quota | 20 requests |
| Observed errors | HTTP 429, 503, 504; `ReadTimeout` |

These operational metrics were reconstructed from terminal logs after persistent audit logging was found to be incomplete; subsequent audit logging was corrected and verified. Their preliminary status limits their use as release evidence.

Implementation checks include **71/71 passing unit tests**, successful LightGBM–Gemini integration and LangGraph workflow execution, verified structured JSON output, retry and timeout handling, persistent audit logging, and safe deferral on API failure. The B0 F2 sandbox repair passed validation, and no live data modification was recorded. **B2 F2 sandbox repair, the updated Gemini prompt verification, and full benchmark evaluation remain pending.**

### 3.4 Promotion Thresholds

| Gate | Acceptance criterion | Current assessment |
|---|---|---|
| **Diagnosis quality** | Previously specified requirement: **at least 76.6% accuracy and 15 percentage points above B0** on matched cases | B1 meets the absolute threshold but improves on B0 by only **5.25 percentage points**. With B0 at 94.75%, a further 15-point gain is mathematically unattainable. The relative-improvement criterion requires formal revision before use as a release gate. |
| **Per-class performance** | At least 50 evaluated examples for each F1–F6 class, with class-specific confusion and recall reporting | Historical benchmark results available; generalization to a fresh evaluation set remains unverified. |
| **Repair effectiveness** | At least **60%** of representative sandbox repair attempts pass predefined post-repair checks | Not established. B1 generated **400 proposals/no-action decisions**, not 400 executed repairs; one F2 sandbox example passed. |
| **Repair safety** | Harmful repair rate no greater than **2%**, with zero unauthorized live writes | Not established on a sufficiently sized repair test set. Observing no live modifications in limited demonstrations does not establish a measured harmful-repair rate. |
| **Operational reliability** | Passing safety tests and acceptable error, deferral, and latency results over the required sample | Unit tests pass, but B2's **42.86% deferral rate** and incomplete repair verification prevent promotion. |

**Promotion disposition:** B0 remains the controlled reference and fallback. B1 remains eligible for further offline and supervised evaluation once the inconsistent accuracy gate is formally resolved and repair-safety evidence is sufficient. B2 remains experimental pending broader evaluation, API version verification, lower failure rates, and completed safety testing. **Autonomous production repair is not authorized for any architecture.**

## 4. Rollback Plan and Rehearsal

**Rollback owner:** Designated experiment operator.  
**Independent reviewer:** Second project team member.  
**Rollback target:** Week 8 B0 rulebook with compatible configuration; otherwise, AI-assisted repair disabled.  
**Historical baseline reference:** `fc47e8c41c7446fdb402b328f42e78939da8e564` (the exact full-test working-tree state associated with this earlier reference was not fully verified).

**Rollback triggers:** Failed promotion gate; unsafe or unauthorized proposed action; failed sandbox postcondition; material diagnosis error; unsupported input schema; API failure or excessive deferral; or incompatible changes to model, thresholds, preprocessing, prompt, or dataset version.

### Rollback Procedure

1. **Suspend candidate output and repair execution.** Stop the experimental workflow and block release of new candidate recommendations. Preserve incident inputs and the last validated batch outputs.
2. **Capture the starting state.** Record `git rev-parse HEAD`, `git status --short`, the DVC dataset pointer/hash, B0 configuration, B1 model manifest and feature schema, and B2 model/prompt identifiers where applicable.
3. **Restore the baseline evaluation path.** Select the B0 rules and compatible feature pipeline. The documented local verification command is:

   ```powershell
   python scripts/evaluate_b0.py --data-dir data/benchmark_1200_v1 --config configs/b0_rules.json --splits test --output results/b0_rollback_metrics.json
   ```

   Execution requires access to the versioned benchmark data. When that dataset is unavailable, the credential-free fixture workflow can be checked with `python scripts/smoke_test.py`. These commands verify a baseline evaluation path; they do **not** constitute a production traffic switch.
4. **Verify known behavior.** Confirm the baseline's permitted labels (`CLEAN`, F1–F6), constrained runbook outputs, and recorded evaluation result on the matching benchmark (**379/400 correct diagnoses**). Verify the actual output file and case counts.
5. **Confirm safety and compatibility.** Maintain human approval and sandbox restrictions. Verify that no live dataset was modified. In any future blue/green batch deployment, restore the previously validated batch output along with its compatible preprocessing and configuration.
6. **Record and review the event.** Document elapsed time, manual step count, results, observed problems, artifact locations, and operator/reviewer approvals. A failed rehearsal blocks promotion until the cause is resolved.

### Local B0 Fallback Verification — November 24, 2026

A timed **local verification of the B0 fallback evaluation path** was performed against the previously used 400-case benchmark. The B0 evaluation script exited successfully and reproduced the known baseline metrics. This is evidence of B0 availability and reproducibility, **not evidence of a completed B1/B2-to-B0 live serving switch**: no active candidate router was switched, and the verification script does not execute repairs or validate sandbox safety.

| Recorded field | Observed value |
|---|---|
| Date and recorded start time | **2026-11-24, 12:54:54** (local PowerShell clock) |
| Elapsed execution time | **0.412 seconds**, measured by `System.Diagnostics.Stopwatch`; includes B0 command execution and PowerShell overhead before stopping the timer |
| Starting branch / commit | `feature/week11-results` / `f73a49c3c2f79ae779cef1bcf0fca84c47587ea7` |
| Baseline being verified | B0 rulebook using `scripts/evaluate_b0.py` and `configs/b0_rules.json` |
| Input and cases | `data/benchmark_1200_v1/test.csv`; **400** records |
| Input SHA-256 | `59B29E1EB312BEACA929262578EFF8B7E55FD8FBF14D1D14D1897A3B052182B7` |
| B0 config SHA-256 | `3E0A29AC065EAB2B9CA55E0CA7E9AB81520880EC42D2120E9102A1DA3DB79543` |
| Starting B1 model version / B2 provider version | **Not captured**; no candidate was actively serving during this test |
| Manual steps | **Not formally counted**; one B0 evaluation command was executed after manual evidence-directory setup and state capture |
| Independent reviewer | **Not recorded; independent verification outstanding** |
| Exit code | **0 (success)** |
| Output artifact on the testing machine | `results/release_rollback_20261124_125454/b0_metrics.json` (local `results/` folder is Git-ignored) |
| Known baseline diagnosis | **379/400 (94.75%)**, reproduced |
| Known detection precision / recall | **100% / 93%**, reproduced |
| Per-class results | `CLEAN`, F1, F2, F3 and F6: **100%**; F4: **80%**; F5: **78%**, reproduced |
| Deferrals / repair execution | **0** deferrals; **no repair execution or sandbox validation performed** by this command |

**Assessment:** **PASS — B0 local fallback-path verification. PARTIAL — full rollback rehearsal.** The remaining release-readiness evidence is a documented switch from an explicitly running B1/B2 candidate to B0 or AI-off, with the complete manual step count, starting model/configuration version, independent reviewer verification and confirmation that the repair execution path remains disabled. No production routing mechanism is claimed.

## 5. Operator Communication and Correction

The release process incorporates the applicable human–AI interaction guidelines:

- **G18 — Notify users about changes.** Candidate outputs must identify B0, B1, or B2; disclose material capability changes, uncertainty, deferrals, and the status of sandbox validation. B1 and B2 must be identified as experimental during supervised demonstrations.
- **G9 — Support correction.** Operators must be able to reject, edit, or escalate incorrect diagnoses and repair proposals. Uncertain evidence and failed API calls must result in manual review or a verified fallback, not automatic execution.
- **G14 — Update and adapt cautiously.** Changes to models, input features, thresholds, prompts, and preprocessing must pass staged validation and preserve the prior validated configuration.

These requirements govern release readiness; implementation of every operator-facing control has not been established by the existing unit and sandbox tests.

## 6. Scope Restrictions and Operational Monitoring

**Release restrictions.** Only authorized public NYC TLC samples and synthetic PipeFail incidents may be used. Data sent to a hosted model must be minimized and explicitly approved. Identifying details, credentials, and private enterprise records must not be exposed. The project retains its **December 31, 2026** raw-data deletion deadline. Repair actions remain restricted, sandbox-validated, and subject to human approval; destructive operations and live autonomous writes are prohibited.

**Monitoring requirements.** Every evaluation batch must capture incident totals, failure-type distributions, false positives/negatives, per-class diagnosis recall, B0–candidate disagreements, API errors, retry counts, deferrals, latency, sandbox outcomes, reviewer decisions, and suspected harmful changes. A designated team member must review these records after each stage and before a promotion decision. Any future operational deployment would additionally require a defined monitoring schedule, thresholds for stopping exposure, and an independently verified rollback mechanism.

## 7. Evidence and Open Items

**Evidence locations**

- `WALKTHROUGH.md` — B0/B1 architecture, 400-case benchmark, runbook proposals, sandbox example, and reproducibility qualifications.
- `MODEL_CARD.md` — B0 model definition, per-class metrics, and limitations.
- `docs/experiments/walkthrough_v1/` — B0/B1 metrics, predictions, provenance, model artifacts, and sandbox audit.
- `docs/b2_results/metrics.json` and `docs/b2_results/README.md` — preliminary B2 validation and operational results.
- `docs/b2_results/f2_mock_summary.json` — mocked F2 verification; not a production repair success measurement.
- `data/benchmark_1200_v1.dvc` — versioned benchmark data pointer.
- CS6170 Week 10, *Deployment Strategies*, slides 8–20 — serving patterns, rollout strategies, promotion gates, rollback, and operator notification requirements.

**Outstanding release prerequisites:**

1. Complete the candidate-to-B0/AI-off switch rehearsal with an explicit starting model version, manual-step count, and independent review. The timed local B0-path verification performed on November 24, 2026 is recorded in Section 4.
2. Verify the exact Gemini API model/version and re-evaluate the updated prompt.
3. Evaluate B2 on a larger, appropriately controlled benchmark and complete the B2 F2 sandbox repair assessment.
4. Conduct fresh per-class and repair-safety evaluation with documented sample counts.
5. Resolve the inconsistent 15-percentage-point improvement criterion through a documented project decision.

**Current release status:** Offline, human-supervised demonstration only. No autonomous production deployment authorized.
