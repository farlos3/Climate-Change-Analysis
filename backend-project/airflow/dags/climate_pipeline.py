import sys
import os
import pandas as pd
import requests
import json

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# เพิ่ม src path เข้า Python path
sys.path.append('/opt/airflow')
sys.path.append('/opt/airflow/src')

# Task functions for Production & Demo Pipeline
def fetch_power_api_task():
    """Fetch climate data from NASA POWER API"""
    from src.ingestion_power import fetch_power_daily_batch
    
    print("🌍 Fetching climate data from NASA POWER API...")
    
    result = fetch_power_daily_batch(
        '/opt/airflow/src/nasa_daily_parameters.csv', 
        '/opt/airflow/data/raw/power_daily.parquet'
    )
    
    print(f'✅ API DATA FETCHED: {result}')
    return result

def smart_load_raw_task():
    """Smart loading: auto-detect fresh start vs incremental update"""
    from src.etl_to_duckdb import smart_load_raw_to_duckdb
    
    print("🧠 Smart loading: auto-detecting fresh start vs incremental update...")
    
    result = smart_load_raw_to_duckdb(
        '/opt/airflow/data/raw/power_daily.parquet', 
        '/opt/airflow/data/duckdb/climate.duckdb',
        'climate_raw'
    )
    
    if result['status'] == 'fresh_start':
        print(f"🆕 FRESH START: Created new table with {result['total_rows']:,} rows")
        print(f"   📅 Date range: {result['date_range']}")
    elif result['status'] == 'incremental_update':
        print(f"🔄 INCREMENTAL UPDATE: {result['total_rows_after']:,} total rows")
        print(f"   ➕ Added: {result['new_rows_added']} | 🔄 Replaced: {result['overlap_rows_replaced']}")
        print(f"   📅 New range: {result['new_data_range']}")
    
    print(f'✅ RAW DATA LOADED: {result["status"]}')
    return result

def prepare_clean_data_task():
    """
    Data preparation: Raw parquet → Clean parquet
    ทำ data cleaning และ preparation
    """
    from src.data_preparation import prepare_nasa_power_data
    print("🧹 DATA PREPARATION: Processing raw data...")
    raw_path = "/opt/airflow/data/raw/power_daily.parquet"
    output_clean_path = "/opt/airflow/data/prepared/climate_clean.parquet"
    # Always prepare new data, overwrite clean parquet
    df_clean = prepare_nasa_power_data(
        raw_parquet_path=raw_path,
        output_parquet_path=output_clean_path,
        quality_checks=True,
    )
    # ให้แน่ใจว่า column วันที่เป็น datetime และชื่อ 'DATE'
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
    print("✅ DATA PREPARATION COMPLETED:")
    print(f"   📊 Final rows: {result['final_rows']:,}")
    print(f"   📅 Range: {result['date_range']}")
    print(f"   🔧 Operation: {result['operation']}")
    return result


def load_clean_to_duckdb_task():
    """
    Load prepared parquet → DuckDB
    เฉพาะการ load ข้อมูลที่ prepare แล้ว
    """
    from src.etl_to_duckdb import load_prepared_to_duckdb_direct
    print("📥 LOADING CLEAN DATA: Prepared parquet → DuckDB...")
    result = load_prepared_to_duckdb_direct(
        prepared_parquet_path='/opt/airflow/data/prepared/climate_clean.parquet',
        duckdb_path='/opt/airflow/data/duckdb/climate.duckdb',
        table_name='climate_clean'
    )
    print(f"✅ CLEAN DATA LOADED:")
    print(f"   📊 Loaded: {result['loaded_rows']:,} rows")
    print(f"   📅 Range: {result['data_range']}")
    print(f"   📥 Operation: {result['operation']}")
    return result

def feature_engineering_task():
    from src.feature_engineering import engineer_t2m_features_from_duckdb
    print("🧑‍🔬 FEATURE ENGINEERING: Generating features from DuckDB...")
    duckdb_path = '/opt/airflow/data/duckdb/climate.duckdb'
    table_name = 'climate_clean'
    output_path = '/opt/airflow/data/prepared/feature_engineering_t2m.parquet'
    df_fe, feature_cols = engineer_t2m_features_from_duckdb(
        duckdb_path=duckdb_path,
        table_name=table_name,
        output_path=output_path
    )
    print(f"✅ FEATURE ENGINEERING COMPLETED: {df_fe.shape[0]:,} rows, {len(feature_cols)} features")
    print(f"   📤 Saved to: {output_path}")
    # --- Save features to DuckDB table ---
    from src.etl_to_duckdb import load_features_to_duckdb
    features_table_name = 'climate_clean'
    duckdb_result = load_features_to_duckdb(
        features_file_path=output_path,
        duckdb_path=duckdb_path,
        table_name=features_table_name
    )
    print(f"✅ Features saved to DuckDB table: {features_table_name}")
    print(f"   📊 Loaded: {duckdb_result['loaded_rows']:,} rows")
    print(f"   📅 Range: {duckdb_result['data_range']}")
    return output_path

# Legacy wrapper สำหรับ backward compatibility
def load_prepared_to_duckdb_task():
    """Legacy wrapper - now uses new data preparation flow"""
    return load_clean_to_duckdb_task()

def notify_backend_features_task(**context):
    """
    ให้ Airflow ยิงไปหา backend API บอกว่า
    'เฮ้ feature ใหม่พร้อมแล้วนะ'
    """

    backend_url = "http://fastapi:8000/ingest/features"
    print(f"📡 Notifying backend at {backend_url}")
    

    # ดึงข้อมูลจาก DuckDB table climate_features
    import duckdb
    duckdb_path = "/opt/airflow/data/duckdb/climate.duckdb"
    table_name = "climate_clean"
    con = duckdb.connect("md:Climate Change (T2M)") 
    df = con.execute(f"SELECT * FROM {table_name}").df()
    con.close()

    # แปลง datetime/Timestamp เป็น string เพื่อให้ serialize เป็น JSON ได้
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    features_json = df.to_dict(orient="records")

    payload = {
        "source": "airflow_climate_pipeline",
        "row_count": len(df),
        "features": features_json,
    }

    resp = requests.post(backend_url, json=payload, timeout=30)
    print("Backend status:", resp.status_code)
    print("Backend response:", resp.text)

    resp.raise_for_status()  # ให้ task fail ถ้า backend 4xx/5xx

    return resp.json()

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
    schedule="@daily",  # Production: daily schedule
    catchup=False,
    tags=["climate", "nasa", "production", "demo"],
    max_active_runs=1,
    max_active_tasks=1
) as dag:

    # Production Tasks
    ingest_task = PythonOperator(
        task_id="fetch_power_api",
        python_callable=fetch_power_api_task,
        retries=2,  # API calls need retry
    )

    smart_load_raw = PythonOperator(
        task_id="smart_load_raw_data",
        python_callable=smart_load_raw_task,
        retries=1,
    )

    prepare_clean = PythonOperator(
        task_id="prepare_clean_data", 
        python_callable=prepare_clean_data_task,
        retries=1,
    )

    load_clean = PythonOperator(
        task_id="load_clean_to_duckdb",
        python_callable=load_clean_to_duckdb_task,
        retries=1,
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering_task",
        python_callable=feature_engineering_task,
        retries=1,
    )
    
    notify_backend = PythonOperator(
        task_id="notify_backend_features",
        python_callable=notify_backend_features_task,
        retries=1,
    )
    
    ingest_task >> smart_load_raw >> prepare_clean >> load_clean >> feature_engineering >> notify_backend