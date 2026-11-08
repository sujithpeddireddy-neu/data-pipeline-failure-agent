import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from pipefail.repair.sandbox import SandboxResult
from pipefail.repair.validator import validate_sandbox


@pytest.fixture
def case(tmp_path):
    values = [
        pa.array([2.5, 3.5, 4.5], type=pa.float64()),
        pa.array([10.0, 20.0, 30.0], type=pa.float64()),
    ]

    reference_schema = pa.schema([
        pa.field("trip_distance", pa.float64(), nullable=False),
        pa.field("fare_amount", pa.float64(), nullable=True),
    ])

    repaired_schema = pa.schema([
        pa.field("trip_distance", pa.float64(), nullable=True),
        pa.field("fare_amount", pa.float64(), nullable=True),
    ])

    reference = tmp_path / "reference.parquet"
    repaired = tmp_path / "repaired.parquet"

    pq.write_table(
        pa.Table.from_arrays(values, schema=reference_schema),
        reference,
    )

    pq.write_table(
        pa.Table.from_arrays(values, schema=repaired_schema),
        repaired,
    )

    sandbox = SandboxResult(
        "test-f2", "SUCCEEDED", str(repaired)
    )

    return reference, repaired, sandbox


def contract(nullable=True):
    return {
        "name": "trip_distance",
        "type": "double",
        "nullable": nullable,
    }


def validate(case, **kwargs):
    reference, _, sandbox = case
    return validate_sandbox(sandbox, reference, **kwargs)


def test_strict_validation_rejects_schema_difference(case):
    result = validate(case)
    assert result.status == "FAILED"


def test_trusted_contract_accepts_correct_repair(case):
    result = validate(
        case,
        expected_restored_field=contract(),
        schema_contract_id="training-v1",
    )
    assert result.status == "PASSED"
    assert result.passed


def test_contract_requires_identifier(case):
    result = validate(
        case,
        expected_restored_field=contract(),
    )
    assert result.status == "FAILED"


def test_contract_rejects_incorrect_nullability(case):
    result = validate(
        case,
        expected_restored_field=contract(nullable=False),
        schema_contract_id="training-v1",
    )
    assert result.status == "FAILED"


def test_contract_does_not_allow_modified_values(case):
    reference, repaired, sandbox = case

    table = pq.read_table(repaired)
    index = table.schema.get_field_index("fare_amount")

    modified = table.set_column(
        index,
        table.schema.field(index),
        pa.array([999.0, 20.0, 30.0]),
    )

    pq.write_table(modified, repaired)

    result = validate(
        case,
        expected_restored_field=contract(),
        schema_contract_id="training-v1",
    )

    assert result.status == "FAILED"


def test_contract_does_not_override_unrelated_columns(case):
    reference, repaired, sandbox = case

    table = pq.read_table(repaired)
    index = table.schema.get_field_index("fare_amount")

    modified = table.set_column(
        index,
        table.schema.field(index).with_nullable(False),
        table.column(index),
    )

    pq.write_table(modified, repaired)

    result = validate(
        case,
        expected_restored_field=contract(),
        schema_contract_id="training-v1",
    )

    assert result.status == "FAILED"
