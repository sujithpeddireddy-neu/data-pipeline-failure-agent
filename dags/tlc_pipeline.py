"""
CS6170 Data Pipeline Failure Agent

NYC TLC Yellow Taxi data-processing DAG.

Pipeline:
1. Ingest monthly Parquet
2. Run data-quality checks
3. Quarantine invalid records
4. Generate DuckDB analytical outputs
"""

import os

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


DATA_DIR = Path(
    os.environ.get(
        "PIPEFAIL_DATA_DIR",
        "/opt/airflow/datasets",
    )
)


def run_ingestion(month: str):
    from pipefail.pipeline.ingest import ingest_month

    return ingest_month(month, DATA_DIR)


def run_quality_checks(month: str):
    from pipefail.pipeline.quality_checks import check_quality

    return check_quality(month, DATA_DIR)


def run_quarantine(month: str):
    from pipefail.pipeline.quarantine import quarantine_month

    return quarantine_month(month, DATA_DIR)


def run_transformations(month: str):
    from pipefail.pipeline.transform import transform_month

    return transform_month(month, DATA_DIR)


with DAG(
    dag_id="tlc_monthly_pipeline",
    description="NYC TLC monthly data pipeline for CS6170",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["CS6170", "TLC", "Data Pipeline"],
) as dag:

    ingest = PythonOperator(
        task_id="ingest_monthly_data",
        python_callable=run_ingestion,
        op_kwargs={
            "month": "{{ dag_run.conf.get('month', '2019-01') if dag_run else '2019-01' }}",
        },
    )

    quality = PythonOperator(
        task_id="validate_data_quality",
        python_callable=run_quality_checks,
        op_kwargs={
            "month": "{{ dag_run.conf.get('month', '2019-01') if dag_run else '2019-01' }}",
        },
    )

    quarantine = PythonOperator(
        task_id="quarantine_invalid_rows",
        python_callable=run_quarantine,
        op_kwargs={
            "month": "{{ dag_run.conf.get('month', '2019-01') if dag_run else '2019-01' }}",
        },
    )

    transform = PythonOperator(
        task_id="generate_analytical_outputs",
        python_callable=run_transformations,
        op_kwargs={
            "month": "{{ dag_run.conf.get('month', '2019-01') if dag_run else '2019-01' }}",
        },
    )

    ingest >> quality >> quarantine >> transform