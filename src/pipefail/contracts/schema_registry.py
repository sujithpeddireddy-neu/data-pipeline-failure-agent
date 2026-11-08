"""Frozen training-only schema contracts, independent of incident reference files.

The trusted source is an original, pre-injection Parquet dataset that must be
reviewed by the team. A contract does NOT certify that the publisher's schema
is universally valid. Never derive a contract from a held-out incident's
baseline.parquet (the answer used for evaluation).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MONTH_RE = re.compile(r"20\d{2}-(0[1-9]|1[0-2])\Z")
_TRAINING_CUTOFF = "2022-12"


def _check_training_month(month: str) -> None:
    if not isinstance(month, str) or not _MONTH_RE.fullmatch(month):
        raise ValueError("month must be YYYY-MM")
    if month < "2019-01" or month > _TRAINING_CUTOFF:
        raise ValueError("Only 2019-2022 training months can seed B0 training contracts")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SchemaContract:
    contract_id: str
    month: str
    source_sha256: str
    fields: dict[str, dict[str, Any]]
    provenance: str

    def verify_field(self, column: str, expected_type: str) -> bool:
        """True only when the expected type is already in the frozen contract."""
        field = self.fields.get(column)
        return bool(field and field.get("type") == expected_type)

    def nullable_for(self, column: str, expected_type: str) -> bool:
        if not self.verify_field(column, expected_type):
            raise ValueError("Proposed target type/column does not match frozen contract")
        nullable = self.fields[column].get("nullable")
        if type(nullable) is not bool:
            raise ValueError("Frozen contract has invalid nullability")
        return nullable


def build_training_contract(*, source: Path, month: str) -> dict[str, Any]:
    """Capture only an approved original *training-month* raw Parquet header.

    Caller must independently check source provenance before freezing the file.
    This reads no incident, features, synthetic label, or clean test reference.
    """
    _check_training_month(month)
    source = Path(source)
    expected_name = f"yellow_tripdata_{month}.parquet"
    if source.name != expected_name or not source.is_file():
        raise ValueError(f"Expected existing original raw file named {expected_name}")
    import pyarrow.parquet as pq

    schema = pq.read_schema(source)
    fields = {
        f.name: {"type": str(f.type), "nullable": bool(f.nullable)}
        for f in schema
    }
    if not fields or "trip_distance" not in fields:
        raise ValueError("Source is missing the expected TLC trip_distance field")
    return {
        "schema_contract_format": 1,
        "contract_id": f"tlc-yellow-{month}-raw-training-v1",
        "month": month,
        "source_file_name": expected_name,
        "source_sha256": sha256_file(source),
        "source_size_bytes": source.stat().st_size,
        "provenance": "Original TLC monthly raw Parquet, captured before synthetic injection; team must verify source provenance.",
        "scope": "TRAINING_PROTOTYPE_ONLY",
        "fields": fields,
    }


def write_contract(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Contract already exists and is frozen: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_contract(path: Path) -> SchemaContract:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_contract_format") != 1 or raw.get("scope") != "TRAINING_PROTOTYPE_ONLY":
        raise ValueError("Unsupported or untrusted schema contract")
    month = raw.get("month")
    _check_training_month(month)
    if raw.get("contract_id") != f"tlc-yellow-{month}-raw-training-v1":
        raise ValueError("Contract ID does not match month")
    digest = raw.get("source_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Missing or invalid source hash")
    fields = raw.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Contract fields missing")
    for field in fields.values():
        if (not isinstance(field, dict)
                or not isinstance(field.get("type"), str)
                or type(field.get("nullable")) is not bool):
            raise ValueError("Invalid field specification in contract")
    return SchemaContract(raw["contract_id"], month, digest, fields, raw.get("provenance", ""))
