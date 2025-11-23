import sys
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# เพิ่ม src path เข้า Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

# Task functions for PythonOperator
def fetch_power_api_task():
    """Fetch climate data from NASA POWER API"""
    from src.ingestion_power import fetch_power_daily_batch
    result = fetch_power_daily_batch(
        '/opt/airflow/src/nasa_daily_parameters.csv', 
        '/opt/airflow/data/raw/power_daily.parquet'
    )
    print(f'SUCCESS: {result}')
    return result

def load_raw_to_duckdb_task():
    """Load raw data to DuckDB"""
    from src.etl_to_duckdb import load_raw_to_duckdb
    result = load_raw_to_duckdb(
        '/opt/airflow/data/raw/power_daily.parquet', 
        '/opt/airflow/data/duckdb/climate.duckdb'
    )
    print(f'SUCCESS: {result}')
    return result


def load_prepared_to_duckdb_task():
    """Load prepared data to DuckDB"""
    from src.etl_to_duckdb import load_prepared_to_duckdb
    result = load_prepared_to_duckdb(
        '/opt/airflow/data/prepared/climate_prepared.parquet', 
        '/opt/airflow/data/duckdb/climate.duckdb'
    )
    print(f'SUCCESS: {result}')
    return result

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

    ingest_task = PythonOperator(
        task_id="fetch_power_api",
        python_callable=fetch_power_api_task,
        retries=0,
    )

    load_raw_duckdb = PythonOperator(
        task_id="load_raw_to_duckdb",
        python_callable=load_raw_to_duckdb_task,
        retries=0,
    )

    load_clean_duckdb = PythonOperator(
        task_id="load_clean_to_duckdb",
        python_callable=load_prepared_to_duckdb_task,
        retries=0,
    )

    # --- DAG FLOW ---
    ingest_task >> load_raw_duckdb >> load_clean_duckdb
