import sys
import os
import pandas as pd
import requests
import json
import duckdb

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

# Task functions for Production & Demo Pipeline
def fetch_power_api_task():

    from src.ingestion_power import fetch_power_daily_batch
    
    print("Fetching climate data from NASA POWER API...")
    
    result = fetch_power_daily_batch(
        '/opt/airflow/src/nasa_daily_parameters.csv', 
        '/opt/airflow/data/raw/power_daily.parquet'
    )
    
    print(f'API DATA FETCHED: {result}')
    return result

def smart_load_raw_task():
    from src.etl_to_duckdb import smart_load_raw_to_duckdb
    
    print("Smart loading: auto-detecting fresh start vs incremental update...")
    
    result = smart_load_raw_to_duckdb(
        '/opt/airflow/data/raw/power_daily.parquet', 
        '/opt/airflow/data/duckdb/climate.duckdb',
        'climate_raw'
    )
    
    if result['status'] == 'fresh_start':
        print(f"FRESH START: Created new table with {result['total_rows']:,} rows")
        print(f"   Date range: {result['date_range']}")
    elif result['status'] == 'incremental_update':
        print(f"INCREMENTAL UPDATE: {result['total_rows_after']:,} total rows")
        print(f"   Added: {result['new_rows_added']} | 🔄 Replaced: {result['overlap_rows_replaced']}")
        print(f"   New range: {result['new_data_range']}")
    
    print(f'RAW DATA LOADED: {result["status"]}')
    return result

def prepare_clean_data_task():

    from src.data_preparation import prepare_nasa_power_data_from_duckdb
    print("DATA PREPARATION: Processing raw data...")

    output_clean_path = "/opt/airflow/data/prepared/climate_clean.parquet"
    
    # Always prepare new data, overwrite clean parquet
    df_clean = prepare_nasa_power_data_from_duckdb(
        duckdb_path="md:Climate Change (T2M)",
        table_name="climate_raw",
        output_path=output_clean_path,
    )

    if "DATE" in df_clean.columns:
        date_col = "DATE"
    else:
        date_col = "date"
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    result = {
        "status": "success",
        "operation": "prepare_and_overwrite",
        "final_rows": len(df_clean),
        "date_range": f"{df_clean[date_col].min().date()} to {df_clean[date_col].max().date()}",
    }
    print("DATA PREPARATION COMPLETED:")
    print(f"   Final rows: {result['final_rows']:,}")
    print(f"   Range: {result['date_range']}")
    print(f"   Operation: {result['operation']}")
    return result


def load_clean_to_duckdb():
    from src.etl_to_duckdb import load_prepared_to_duckdb_direct
    print("LOADING CLEAN DATA: Prepared parquet → DuckDB...")
    result = load_prepared_to_duckdb_direct(
        prepared_parquet_path='/opt/airflow/data/prepared/climate_clean.parquet',
        duckdb_path='md:Climate Change (T2M)',
        table_name='climate_clean'
    )
    print(f"CLEAN DATA LOADED:")
    print(f"   Loaded: {result['loaded_rows']:,} rows")
    print(f"   Range: {result['data_range']}")
    print(f"   Operation: {result['operation']}")
    return result

def feature_engineering_task():
    from src.feature_engineering import engineer_t2m_features_from_duckdb
    print("FEATURE ENGINEERING: Generating features from DuckDB...")
    table_name = 'climate_clean'
    output_path = '/opt/airflow/data/prepared/feature_engineering_t2m.parquet'
    df_fe, feature_cols = engineer_t2m_features_from_duckdb(
        table_name=table_name,
        duckdb_path='md:Climate Change (T2M)',
        output_path=output_path
    )
    print(f"FEATURE ENGINEERING COMPLETED: {df_fe.shape[0]:,} rows, {len(feature_cols)} features")
    print(f"   Saved to: {output_path}")

    from src.etl_to_duckdb import load_features_to_duckdb
    feature_store_table_name = 'feature_store'
    duckdb_result = load_features_to_duckdb(
        features_file_path=output_path,
        duckdb_path='md:Climate Change (T2M)',
        table_name=feature_store_table_name
    )
    print(f"Features saved to DuckDB feature store table: {feature_store_table_name}")
    print(f"   Loaded: {duckdb_result['loaded_rows']:,} rows")
    print(f"   Range: {duckdb_result['data_range']}")
    return output_path

def load_prepared_to_duckdb():
    return load_clean_to_duckdb()

def forecasting(**context):
    from src.predict_process import run_feature_selection, run_prediction, save_forecast_to_duckdb

    print("Running prediction on pipeline only (not sending to backend)...")
    feature_store_table = "feature_store"
    duckdb_path = "md:Climate Change (T2M)"

    con = duckdb.connect(duckdb_path)
    df = con.execute(f"SELECT * FROM {feature_store_table}").df()
    con.close()

    # Convert datetime/timestamp to string so that it can be serialized to JSON.
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    df_selected = run_feature_selection(df)
    features_json = df_selected.to_dict(orient="records")
    prediction_result = run_prediction(features_json)
    print("Prediction result:")
    print(prediction_result)
    save_forecast_to_duckdb(prediction_result, duckdb_path, table_name="forecast_store")
    return prediction_result

default_args = {
    'owner': 'climate_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10)
}

# Production & Demo Ready Pipeline
with DAG(
    dag_id="climate_pipeline",
    default_args=default_args,
    description="Production climate data pipeline with smart incremental updates",
    schedule="@daily",  
    catchup=False,
    tags=["climate", "nasa", "production", "demo"],
    max_active_runs=1,
    max_active_tasks=1
) as dag:

    # Production Tasks
    ingest = PythonOperator(
        task_id="fetch_power_api",
        python_callable=fetch_power_api_task,
        retries=2,  # API calls need retry
    )

    load_raw = PythonOperator(
        task_id="load_raw_data",
        python_callable=smart_load_raw_task,
        retries=1,
    )

    prepare_clean = PythonOperator(
        task_id="prepare_clean_task", 
        python_callable=prepare_clean_data_task,
        retries=1,
    )

    load_clean = PythonOperator(
        task_id="load_clean_task",
        python_callable=load_clean_to_duckdb,
        retries=1,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering_task",
        python_callable=feature_engineering_task,
        retries=1,
    )
    
    forecasting = PythonOperator(
        task_id="forecasting_task",
        python_callable=forecasting,
        retries=1,
    )
    
    ingest >> load_raw >> prepare_clean >> load_clean >> feature_engineering >> forecasting