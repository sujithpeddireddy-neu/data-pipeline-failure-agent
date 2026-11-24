# B2: LightGBM + Gemini - Preliminary Results

## Architecture

B2 integrates LightGBM with Gemini using LangGraph.
Gemini treats the LightGBM diagnosis as a hypothesis
and can override it based on available evidence.

Repair proposals are controlled by deterministic
safeguards. Production repair execution is not allowed,
and human approval remains required for repairs.

## Configuration

- Requested model ID: gemini-3.5-flash
- Temperature: 0
- Maximum output tokens: 4096
- Maximum API attempts: 3
- Configured request timeout: 30 seconds

The quota error reported gemini-3.6-flash.
The actual backend model identity was not independently verified.

## Preliminary validation results

- Dataset: 14 synthetic smoke validation incidents
- B1 accuracy: 92.86%
- B2 overall accuracy: 50.00%
- B2 non-deferred accuracy: 87.50%
- B2 diagnosis coverage: 57.14%
- Correct B2 diagnoses: 7
- Incorrect B2 diagnoses: 1
- API-related deferrals: 6
- Incidents recovered through retries: 2

## Error analysis

Incident 0005 had ground truth F5.
LightGBM correctly predicted F5, while Gemini predicted F4.
Gemini's explanation correctly identified invalid distance
values but assigned the wrong failure category.

The reconstructed input and saved structured response
are included in incident_0005.json.

## Repair verification

The F2 sandbox integration demonstration passed using
a mocked Gemini response, the real LightGBM model,
LangGraph workflow, repair engine and validator.

- Sandbox execution: SUCCEEDED
- Validation: PASSED
- Human approval: PENDING
- Original candidate and reference: unchanged
- Live data modifications: none
- Real API requests: zero

This is a synthetic integration demonstration, not a
live Gemini end-to-end repair or held-out repair-success test.

## Testing status

- Existing unit tests: 71/71 passed
- API retries: verified
- API failure deferrals: verified
- Persistent audit logging: verified
- Mocked F2 sandbox repair: passed
- Updated prompt live verification: pending
- Live Gemini F2 repair verification: pending

## Limitations

The reported free-tier quota was 20 requests per day.
API errors included 429, 503, 504 and ReadTimeout.

The 14-incident evaluation preceded the updated prompt.
Operational metrics were reconstructed from Terminal logs
because persistent API audit logging was missing during
that evaluation. Audit logging was subsequently fixed.

The exported audits are reconstructed copies with raw
exception text removed. Original local records are unchanged.

These are preliminary synthetic smoke results, not full
benchmark results or evidence of real-world generalization.

Full benchmark evaluation was omitted due to API limitations.
