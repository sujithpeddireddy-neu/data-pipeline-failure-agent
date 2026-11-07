"""Safe, size-limited sandbox trial for B0 F2 type restoration."""

from dataclasses import dataclass
from pathlib import Path
import re

from pipefail.contracts.schemas import RepairAction
from pipefail.repair.actions import RepairProposal, RESTORABLE_TYPES


@dataclass(frozen=True)
class SandboxResult:
    incident_id: str
    status: str  # SUCCEEDED | FAILED | SKIPPED
    output_path: str | None = None
    message: str = ""


def execute_in_sandbox(
    proposal: RepairProposal,
    candidate_path: str | Path,
    sandbox_dir: str | Path,
    *,
    max_bytes: int = 50 * 1024 * 1024,
) -> SandboxResult:
    """Write ONLY a fresh .parquet under sandbox_dir; never touch source/live data.

    max_bytes is intentionally small for the no-credentials baseline demonstration.
    This is not a streaming repair engine for production-sized Parquet files.
    """
    if proposal.status != "SANDBOX_ELIGIBLE" or proposal.action != RepairAction.CONVERT_TYPE:
        return SandboxResult(proposal.incident_id, "SKIPPED", message="No allowlisted sandbox trial.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", proposal.incident_id):
        return SandboxResult(proposal.incident_id, "FAILED", message="Unsafe incident ID.")

    source = Path(candidate_path).resolve(strict=False)
    root = Path(sandbox_dir).resolve(strict=False)
    if not source.is_file() or source.suffix.lower() != ".parquet":
        return SandboxResult(proposal.incident_id, "FAILED", message="Candidate Parquet missing.")
    if source.stat().st_size > max_bytes:
        return SandboxResult(proposal.incident_id, "FAILED", message="Input exceeds sandbox size limit.")
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        return SandboxResult(proposal.incident_id, "FAILED", message="Invalid sandbox directory.")
    # Avoid any attempt to write alongside or inside the source data folder.
    if root == source.parent or root in source.parents or source == root:
        return SandboxResult(proposal.incident_id, "FAILED", message="Sandbox overlaps source path.")
    target_path = root / f"{proposal.incident_id}_trial.parquet"
    if target_path.exists() or target_path.is_symlink():
        return SandboxResult(proposal.incident_id, "FAILED", message="Trial output already exists.")
    temp_path = root / f"{proposal.incident_id}_trial.parquet.part"
    if temp_path.exists() or temp_path.is_symlink():
        return SandboxResult(proposal.incident_id, "FAILED", message="Temporary output already exists.")
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        col = proposal.parameters["column"]
        src_type = proposal.parameters["source_type"]
        dst_type = proposal.parameters["target_type"]
        if dst_type not in RESTORABLE_TYPES:
            raise ValueError("Disallowed destination type")
        table = pq.read_table(source)
        index = table.schema.get_field_index(col)
        if index < 0 or str(table.schema.field(index).type) != src_type:
            raise ValueError("Candidate schema does not match the proposed preconditions")
        arrow_type = {"double": pa.float64(), "float": pa.float32(),
                      "int64": pa.int64(), "int32": pa.int32()}[dst_type]
        casted = pc.cast(table.column(index), arrow_type, safe=True)
        new_table = table.set_column(index, table.schema.field(index).with_type(arrow_type), casted)
        pq.write_table(new_table, temp_path, compression="zstd")
        # The only created file is in sandbox root.
        temp_path.replace(target_path)
        return SandboxResult(proposal.incident_id, "SUCCEEDED", str(target_path),
                             f"Sandbox trial written; {new_table.num_rows} rows. NOT approved.")
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        return SandboxResult(proposal.incident_id, "FAILED",
                             message=f"Sandbox type restoration failed: {type(exc).__name__}: {exc}")
