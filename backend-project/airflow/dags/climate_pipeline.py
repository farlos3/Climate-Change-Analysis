import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# เพิ่ม src path เข้า Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

from src.ingestion_power import fetch_power_daily_batch
from src.prepare_climate import prepare_climate_data
from src.etl_to_duckdb import load_raw_to_duckdb, load_prepared_to_duckdb

default_args = {
    'owner': 'climate_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30)
}


with DAG(
    dag_id="climate_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
    tags=["climate", "nasa", "duckdb"],
    max_active_runs=1,
    max_active_tasks=1
) as dag:

    ingest_task = BashOperator(
        task_id="fetch_power_api",
        bash_command="""
        cd /opt/airflow && python -c "
import sys
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')
from src.ingestion_power import fetch_power_daily_batch
result = fetch_power_daily_batch('/opt/airflow/src/nasa_daily_parameters.csv', '/opt/airflow/data/raw/power_daily.parquet')
print(f'SUCCESS: {result}')
        "
        """,
        retries=0,
    )

    load_raw_duckdb = BashOperator(
        task_id="load_raw_to_duckdb",
        bash_command="""
        cd /opt/airflow && python -c "
import sys
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')
from src.etl_to_duckdb import load_raw_to_duckdb
result = load_raw_to_duckdb('/opt/airflow/data/raw/power_daily.parquet', '/opt/airflow/data/duckdb/climate.duckdb')
print(f'SUCCESS: {result}')
        "
        """,
        retries=0,
    )

    prepare_task = BashOperator(
        task_id="prepare_climate",
        bash_command="""
        cd /opt/airflow && python -c "
import sys
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')
from src.prepare_climate import prepare_climate_data
result = prepare_climate_data('/opt/airflow/data/duckdb/climate.duckdb', '/opt/airflow/data/prepared/climate_prepared.parquet', '/opt/airflow/models/feature_scaler.pkl', 'climate_raw')
print(f'SUCCESS: {result}')
        "
        """,
        retries=0,
    )

    load_clean_duckdb = BashOperator(
        task_id="load_clean_to_duckdb",
        bash_command="""
        cd /opt/airflow && python -c "
import sys
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')
from src.etl_to_duckdb import load_prepared_to_duckdb
result = load_prepared_to_duckdb('/opt/airflow/data/prepared/climate_prepared.parquet', '/opt/airflow/data/duckdb/climate.duckdb')
print(f'SUCCESS: {result}')
        "
        """,
        retries=0,
    )

    # --- DAG FLOW ---
    ingest_task >> load_raw_duckdb >> prepare_task >> load_clean_duckdb

