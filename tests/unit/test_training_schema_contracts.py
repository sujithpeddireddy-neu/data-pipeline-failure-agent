"""Training-only contract tests, including provenance and no-overwrite guards."""

import hashlib
import json

import pytest

from pipefail.contracts.schema_registry import (
    SchemaContract,
    build_training_contract,
    load_contract,
    write_contract,
)


def test_rejects_future_month_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match="training months"):
        build_training_contract(source=tmp_path / "nonexistent.parquet", month="2025-01")


def test_refuses_to_overwrite_frozen_contract(tmp_path):
    path = tmp_path / "frozen.json"
    write_contract({"hello": "world"}, path)
    with pytest.raises(FileExistsError):
        write_contract({"hello": "changed"}, path)


def test_contract_loader_checks_provenance_and_types(tmp_path):
    p = tmp_path / "contract.json"
    doc = {
        "schema_contract_format": 1,
        "scope": "TRAINING_PROTOTYPE_ONLY",
        "contract_id": "tlc-yellow-2019-01-raw-training-v1",
        "month": "2019-01",
        "source_sha256": "1" * 64,
        "fields": {"trip_distance": {"type": "double", "nullable": False}},
        "provenance": "verified original raw file",
    }
    write_contract(doc, p)
    loaded = load_contract(p)
    assert loaded.nullable_for("trip_distance", "double") is False
    with pytest.raises(ValueError, match="target type"):
        loaded.nullable_for("trip_distance", "string")


def test_rejects_invalid_nullability_in_json(tmp_path):
    p = tmp_path / "bad.json"
    doc = {"schema_contract_format": 1, "scope": "TRAINING_PROTOTYPE_ONLY",
           "contract_id": "tlc-yellow-2019-01-raw-training-v1",
           "month": "2019-01", "source_sha256": "a" * 64,
           "fields": {"trip_distance": {"type": "double", "nullable": "false"}}}
    write_contract(doc, p)
    with pytest.raises(ValueError, match="Invalid field"):
        load_contract(p)


def test_build_from_original_parquet_reads_header_and_hash(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "yellow_tripdata_2019-01.parquet"
    table = pa.Table.from_arrays(
        [pa.array([1.0, 2.0], type=pa.float64())],
        schema=pa.schema([pa.field("trip_distance", pa.float64(), nullable=False)]),
    )
    pq.write_table(table, source)
    payload = build_training_contract(source=source, month="2019-01")
    assert payload["fields"]["trip_distance"] == {"type": "double", "nullable": False}
    assert payload["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
