"""Run one B2 example with existing trained LightGBM model."""
import argparse
import json
import os
from pathlib import Path
import pandas as pd
from pipefail.diagnosis.lightgbm_runbook import LightGBMRunbook
from pipefail.diagnosis.gemini_agent import GeminiAgent
from pipefail.workflows.gemini_workflow import run_gemini_workflow

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--model",default=os.getenv("GEMINI_MODEL"))
    p.add_argument("--split",choices=("train","val","test"),default="val")
    p.add_argument("--row",type=int,default=0)
    p.add_argument("--evidence-json",type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("results/gemini_demo"))
    a=p.parse_args()
    if not a.model:
        p.error("Set GEMINI_MODEL or --model to an exact Gemini model ID")
    rec=LightGBMRunbook(a.model_dir/"model.txt",a.model_dir/"run_manifest.json")
    df=pd.read_csv(a.data_dir/f"{a.split}.csv")
    if a.row<0 or a.row>=len(df):
        p.error("Row out of range")
    row=df.iloc[a.row].to_dict()
    raw=json.loads(a.evidence_json.read_text()) if a.evidence_json else {}
    result=run_gemini_workflow(
        recommender=rec,gemini=GeminiAgent(a.model),
        incident_id=f"b2-{a.split}-{a.row:04d}",
        features={k:row[k] for k in rec.feature_names},
        raw_evidence=raw,output_dir=a.output_dir)
    print(json.dumps({"lightgbm":result["lightgbm"],"gemini":result["gemini"],
                      "proposal":result["proposal"]["action"],
                      "review_status":result["review_status"]},indent=2))
if __name__=="__main__":
    main()
