"""
CS6170 Data Pipeline Failure Agent
Initial Apache Airflow environment verification.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def verify_environment():
    """Verify that Airflow can execute Python tasks."""

    print("CS6170 Data Pipeline Failure Agent")
    print("Apache Airflow is running successfully!")
    print("Python task execution verified.")


with DAG(
    dag_id="pipefail_smoke_test",
    description="Initial environment verification for CS6170",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["CS6170", "setup"],
) as dag:

    verify_task = PythonOperator(
        task_id="verify_environment",
        python_callable=verify_environment,
    )