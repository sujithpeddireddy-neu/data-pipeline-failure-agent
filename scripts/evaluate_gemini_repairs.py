"""Small B1/B2 comparison; diagnosis only, no production repair."""
import argparse
import csv
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
    p.add_argument("--limit",type=int,default=10)
    p.add_argument("--output-dir",type=Path,default=Path("results/gemini_evaluation"))
    a=p.parse_args()
    if not a.model or a.limit<1:
        p.error("Specify Gemini model and positive limit")
    rec=LightGBMRunbook(a.model_dir/"model.txt",a.model_dir/"run_manifest.json")
    agent=GeminiAgent(a.model)
    df=pd.read_csv(a.data_dir/f"{a.split}.csv").head(a.limit)
    rows=[]
    for i,row in enumerate(df.to_dict("records")):
        incident=f"b2-{a.split}-{i:04d}"
        features={k:row[k] for k in rec.feature_names}
        b1,_=rec.recommend(features,incident)
        b1label="CLEAN" if b1.failure_type.value=="NONE" else b1.failure_type.value
        result=run_gemini_workflow(recommender=rec,gemini=agent,
                                   incident_id=incident,features=features,
                                   output_dir=a.output_dir/"audits")
        b2label=result["diagnosis"]["failure_type"]
        if b2label=="NONE":
            b2label="CLEAN"
        rows.append({"incident":incident,"true_label":row["label"],
                     "b1_label":b1label,"b2_label":b2label,
                     "b1_correct":b1label==row["label"],
                     "b2_correct":b2label==row["label"],
                     "b2_deferred":b2label=="UNKNOWN",
                     "review_status":result["review_status"]})
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/"comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("Cases:",len(rows))
    print("B1 accuracy:",sum(r["b1_correct"] for r in rows)/len(rows))
    print("B2 accuracy:",sum(r["b2_correct"] for r in rows)/len(rows))
    print("B2 deferrals:",sum(r["b2_deferred"] for r in rows))
    print("No repair success measured; test split may have been previously inspected.")
if __name__=="__main__":
    main()
