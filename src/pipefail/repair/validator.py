"""Compare sandbox output against an independent clean reference (not just self-checks)."""

from dataclasses import dataclass, field
from pathlib import Path

from pipefail.repair.sandbox import SandboxResult


@dataclass(frozen=True)
class ValidationResult:
    incident_id: str
    status: str  # PASSED | FAILED | NOT_EVALUATED
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)


def validate_sandbox(
    sandbox: SandboxResult,
    clean_reference_path: str | Path | None,
    *,
    max_bytes: int = 50 * 1024 * 1024,
    expected_restored_field: dict | None = None,
    schema_contract_id: str | None = None,
) -> ValidationResult:
    """PASSED only for exact matching schemas, rows and cell values on a clean reference.

    No reference == no proof of safety. This intentionally does NOT validate
    preservation of downstream business semantics beyond the clean reference.
    """
    restored = expected_restored_field
    if restored is not None:
        if (not schema_contract_id or not isinstance(restored, dict)
                or set(restored) != {"name", "type", "nullable"}
                or not isinstance(restored["name"], str) or not restored["name"]
                or not isinstance(restored["type"], str) or not restored["type"]
                or type(restored["nullable"]) is not bool):
            return ValidationResult(sandbox.incident_id, "FAILED", False,
                                    ("Invalid trusted schema contract.",))

    if sandbox.status != "SUCCEEDED" or not sandbox.output_path:
        return ValidationResult(sandbox.incident_id, "NOT_EVALUATED", False,
                                ("No completed sandbox trial to validate.",))
    if not clean_reference_path:
        return ValidationResult(sandbox.incident_id, "NOT_EVALUATED", False,
                                ("No independent clean-reference file provided.",))
    ref = Path(clean_reference_path)
    out = Path(sandbox.output_path)
    if not ref.is_file() or not out.is_file():
        return ValidationResult(sandbox.incident_id, "NOT_EVALUATED", False,
                                ("Clean reference or sandbox output is missing.",))
    if ref.stat().st_size > max_bytes or out.stat().st_size > max_bytes:
        return ValidationResult(sandbox.incident_id, "NOT_EVALUATED", False,
                                ("Validation sample exceeds size limit.",))
    try:
        import pyarrow.parquet as pq
        reference = pq.read_table(ref)
        candidate = pq.read_table(out)
        if reference.num_rows != candidate.num_rows:
            return ValidationResult(sandbox.incident_id, "FAILED", False,
                                    (f"Row count mismatch {reference.num_rows} != {candidate.num_rows}",))
        if reference.schema.names != candidate.schema.names:
            return ValidationResult(sandbox.incident_id, "FAILED", False,
                                    ("Column names/order changed.",))
        if restored and restored["name"] not in reference.schema.names:
            return ValidationResult(sandbox.incident_id, "FAILED", False,
                                    ("Trusted restored column absent from reference.",))
        for col in reference.schema.names:
            a, b = reference.schema.field(col), candidate.schema.field(col)
            contract_col = restored is not None and col == restored["name"]
            schema_invalid = (
                a.type != b.type
                or (not contract_col and a.nullable != b.nullable)
                or (contract_col and (
                    str(a.type) != restored["type"]
                    or b.nullable != restored["nullable"]
                ))
            )
            if schema_invalid:
                return ValidationResult(sandbox.incident_id, "FAILED", False,
                                        (f"Column schema mismatch: {col} ({a} vs {b})",))
            if not reference.column(col).equals(candidate.column(col)):
                return ValidationResult(sandbox.incident_id, "FAILED", False,
                                        (f"Reference values differ in column {col}.",))
        return ValidationResult(sandbox.incident_id, "PASSED", True,
                                (f"Matched clean reference for all {reference.num_rows} rows and {len(reference.schema.names)} columns.",
                                 "Only this isolated sample was validated; production release is prohibited."))
    except Exception as exc:
        return ValidationResult(sandbox.incident_id, "FAILED", False,
                                (f"Validation error: {type(exc).__name__}: {exc}",))
