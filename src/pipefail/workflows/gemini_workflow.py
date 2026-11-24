"""B2 LightGBM + Gemini + existing conservative repair safeguards."""
from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pipefail.contracts.schemas import DiagnosisResult, FailureType, RepairAction
from pipefail.diagnosis.runbook import DEFAULT_ACTIONS
from pipefail.repair.actions import propose_repair
from pipefail.repair.sandbox import execute_in_sandbox
from pipefail.repair.validator import validate_sandbox

class State(TypedDict, total=False):
    incident_id: str
    features: dict
    raw_evidence: dict
    lightgbm: dict
    gemini: dict
    diagnosis: object
    proposal: object
    error: str

def run_gemini_workflow(*,recommender,gemini,incident_id,features,output_dir,
                        raw_evidence=None,candidate_parquet=None,
                        clean_reference_parquet=None,trusted_nullability=None,
                        schema_contract_id=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}",incident_id):
        raise ValueError("Invalid incident ID")
    output=Path(output_dir)
    audit_file=output/f"{incident_id}_audit.json"
    if audit_file.exists() or audit_file.is_symlink():
        raise FileExistsError(audit_file)
    if trusted_nullability is not None and (
        not schema_contract_id or any(type(v) is not bool for v in trusted_nullability.values())
    ):
        raise ValueError("Trusted nullability requires a valid schema contract")
    raw=dict(raw_evidence or {})
    # Do not send code, raw rows, credentials or arbitrary payload fields.
    safe={k:raw[k] for k in ("reference","candidate","schema_diff","column_statistics","logs") if k in raw}
    if len(json.dumps(safe,default=str))>12000:
        raise ValueError("Sanitized evidence too large")

    def predict(s):
        diagnosis,_=recommender.recommend(s["features"],s["incident_id"])
        label="CLEAN" if diagnosis.failure_type==FailureType.NONE else diagnosis.failure_type.value
        return {"lightgbm":{"label":label,"score_uncalibrated":diagnosis.confidence}}
    def reason(s):
        try:
            answer=gemini.diagnose({"lightgbm":s["lightgbm"],
                                   "features":s["features"],"evidence":safe})
            return {"gemini":answer}
        except Exception as exc:
            return {"error":f"{type(exc).__name__}: {exc}",
                    "gemini":{"failure_type":"UNKNOWN",
                              "recommended_action":"MANUAL_REVIEW",
                              "affected_column":None,"reason":"Gemini unavailable",
                              "evidence":[],"uncertainty":"Human diagnosis required"}}
    def propose(s):
        answer=s["gemini"]
        failure=FailureType(answer["failure_type"])
        action=DEFAULT_ACTIONS.get(failure,RepairAction.MANUAL_REVIEW)
        diagnosis=DiagnosisResult(
            incident_id=s["incident_id"],model_name="LIGHTGBM_GEMINI",
            failure_type=failure,affected_column=answer["affected_column"],
            recommended_action=action,evidence=[answer["reason"],*answer["evidence"]],
            uncertainty=answer["uncertainty"],confidence=None)
        return {"diagnosis":diagnosis,"proposal":propose_repair(diagnosis,raw)}
    graph=StateGraph(State)
    graph.add_node("lightgbm",predict)
    graph.add_node("gemini",reason)
    graph.add_node("propose",propose)
    graph.add_edge(START,"lightgbm")
    graph.add_edge("lightgbm","gemini")
    graph.add_edge("gemini","propose")
    graph.add_edge("propose",END)
    s=graph.compile().invoke({"incident_id":incident_id,"features":dict(features),
                              "raw_evidence":raw})
    proposal=s["proposal"]
    status=("BLOCKED_GEMINI_ERROR" if s.get("error") else
            "NO_ACTION" if s["diagnosis"].failure_type==FailureType.NONE and s["lightgbm"]["label"]=="CLEAN" else
            "PENDING_HUMAN_REVIEW")
    trial=validation=None
    if status=="PENDING_HUMAN_REVIEW" and proposal.status=="SANDBOX_ELIGIBLE" and candidate_parquet:
        col=proposal.parameters["column"]
        nullable=trusted_nullability.get(col) if trusted_nullability else None
        trial=execute_in_sandbox(proposal,candidate_parquet,output/"sandbox",
                                 expected_nullable=nullable)
        field={"name":col,"type":proposal.parameters["target_type"],"nullable":nullable} if nullable is not None else None
        validation=validate_sandbox(trial,clean_reference_parquet,
                                    expected_restored_field=field,
                                    schema_contract_id=schema_contract_id if field else None)
        if trial.status!="SUCCEEDED":
            status="BLOCKED_SANDBOX_FAILED"
        elif validation.status=="FAILED":
            status="BLOCKED_VALIDATION_FAILED"
        elif not validation.passed:
            status="BLOCKED_NOT_VALIDATED"
    d=s["diagnosis"]
    audit={
        "incident_id":incident_id,"architecture":"LIGHTGBM_GEMINI",
        "gemini_model":gemini.model,"lightgbm":s["lightgbm"],"gemini":s["gemini"],
        "api_metrics": {
            "attempts": getattr(gemini, "last_attempts", None),
            "latency_seconds": getattr(gemini, "last_latency_seconds", None),
            "request_errors": getattr(gemini, "last_errors", []),
            "diagnosis_response_success": not bool(s.get("error")),
            "retry_recovered": (
                (getattr(gemini, "last_attempts", 0) or 0) > 1
                and not bool(s.get("error"))
            ),
        },
        "diagnosis":{**asdict(d),"failure_type":d.failure_type.value,
                     "recommended_action":d.recommended_action.value},
        "proposal":{**asdict(proposal),"failure_type":proposal.failure_type.value,
                    "action":proposal.action.value},
        "sandbox":asdict(trial) if trial else None,
        "validation":asdict(validation) if validation else None,
        "review_status":status,"error":s.get("error"),"live_data_modified":False}
    output.mkdir(parents=True,exist_ok=True)
    with audit_file.open("x") as f:
        json.dump(audit,f,indent=2)
    return audit
