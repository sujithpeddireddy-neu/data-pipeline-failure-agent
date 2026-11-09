"""Freeze a training-only TLC schema, with types AND Arrow nullability.

Usage (from repository root):
  python scripts/create_training_schema_contract.py \
      --month 2019-01 \
      --source C:\\Capstone\\datasets\\raw\\yellow_taxi\\yellow_tripdata_2019-01.parquet \
      --output configs/schema_contracts/2019-01.json

Manually verify that --source is an original pre-injection raw TLC file.
The output is a training-prototype contract, NOT an official or production schema.
"""

import argparse
from pathlib import Path

from pipefail.contracts.schema_registry import (
    build_training_contract,
    write_contract,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    contract = build_training_contract(source=args.source, month=args.month)
    write_contract(contract, args.output)
    print("Frozen contract:", args.output)
    print("Source SHA256:", contract["source_sha256"])
    print("Fields:", len(contract["fields"]))
    print("trip_distance:", contract["fields"]["trip_distance"])
    print("Scope: training prototype; NOT an official production contract")


if __name__ == "__main__":
    main()
