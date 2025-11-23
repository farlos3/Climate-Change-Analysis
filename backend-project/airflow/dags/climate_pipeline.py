import sys
import os
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
    import os
    from src.data_preparation import prepare_climate_incremental, prepare_climate_data
    
    print("🧹 DATA PREPARATION: Processing raw data...")
    
    raw_path = '/opt/airflow/data/raw/power_daily.parquet'
    existing_clean_path = '/opt/airflow/data/prepared/climate_clean.parquet' 
    output_clean_path = '/opt/airflow/data/prepared/climate_clean.parquet'
    
    # เช็คว่ามีข้อมูล clean เก่าหรือไม่
    if os.path.exists(existing_clean_path):
        # Incremental preparation
        print("🔄 INCREMENTAL PREPARATION: Adding new data to existing clean data...")
        result = prepare_climate_incremental(
            existing_clean_path=existing_clean_path,
            new_raw_path=raw_path,
            output_clean_path=output_clean_path,
            overlap_days=3  # NASA API 3-day delay
        )
    else:
        # Fresh preparation 
        print("🆕 FRESH PREPARATION: Creating clean data from scratch...")
        result = prepare_climate_data(
            raw_parquet_path=raw_path,
            output_parquet_path=output_clean_path,
            quality_checks=True
        )
    
    print(f"✅ DATA PREPARATION COMPLETED:")
    print(f"   📊 Final rows: {result.get('final_rows', result.get('cleaned_rows', 'N/A')):,}")
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

# Legacy wrapper สำหรับ backward compatibility
def load_prepared_to_duckdb_task():
    """Legacy wrapper - now uses new data preparation flow"""
    return load_clean_to_duckdb_task()

default_args = {
    'owner': 'climate_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30)
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

    # --- PRODUCTION & DEMO FLOW ---
    # สำหรับ Demo: แสดง fresh start ในรันแรก, incremental ในรันถัดไป
    # สำหรับ Production: ทำงานปกติทุกวัน (NASA API delay 3 วัน)
    # Flow: API → Raw → Prepare → Load
    ingest_task >> smart_load_raw >> prepare_clean >> load_clean
