import os
import duckdb
from datetime import datetime, timedelta
from typing import Optional

def smart_load_raw_to_duckdb(
    raw_parquet_path: str, 
    duckdb_path: str, 
    table_name: str = "climate_raw"
) -> dict:
    """
    Smart loading: incremental update หรือ fresh start อัตโนมัติ
    - ถ้าไม่มี table → fresh start
    - ถ้ามี table แล้ว → incremental update with overlap handling
    
    สำหรับ Production และ Demo
    """
    
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Raw parquet not found: {raw_parquet_path}")

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
    con = duckdb.connect(duckdb_path)

    try:
        # ตรวจสอบว่า table มีอยู่แล้วมั้ย
        table_exists = con.execute(f"""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = '{table_name}'
        """).fetchone()[0] > 0
        
        # Always overwrite table with new data (no incremental logic)
        print(f"🆕 OVERWRITE: Creating or replacing {table_name} table...")
        import pandas as pd
        df = pd.read_parquet(raw_parquet_path)
        if 'DATE' in df.columns:
            df = df.rename(columns={'DATE': 'date'})
        if 'date' not in df.columns:
            raise Exception(f"Parquet file '{raw_parquet_path}' does not contain a 'date' column after renaming. Columns found: {list(df.columns)}")
        con.register('raw_df', df)
        col_str = ', '.join(df.columns)
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT {col_str} FROM raw_df
        """)
        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        date_range = con.execute(f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name}").fetchone()
        result = {
            'status': 'overwrite',
            'operation': 'create_or_replace_table',
            'total_rows': row_count,
            'date_range': f"{date_range[0]} to {date_range[1]}",
            'table_name': table_name
        }
        print(f"✅ Overwrite completed: {row_count:,} rows loaded")
        return result
        
    except Exception as e:
        print(f"❌ Error in smart loading: {e}")
        raise e
    finally:
        con.close()

def load_prepared_to_duckdb_direct(
    prepared_parquet_path: str, 
    duckdb_path: str, 
    table_name: str = "climate_clean"
) -> dict:
    """
    Simple loader: อ่าน prepared parquet → DuckDB table
    ไม่ทำ data cleaning หรือ feature engineering อะไรเลย
    เพียงแค่ load ข้อมูลที่ prepared แล้ว
    """
    
    if not os.path.exists(prepared_parquet_path):
        raise FileNotFoundError(f"Prepared parquet not found: {prepared_parquet_path}")
    
    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
    con = duckdb.connect(duckdb_path)
    
    try:
        print(f"📥 LOADING PREPARED DATA: {prepared_parquet_path} → {table_name}")
        
        # อ่าน parquet แล้วแปลง 'DATE' เป็น 'date' ก่อนโหลดเข้า DuckDB
        import pandas as pd
        df = pd.read_parquet(prepared_parquet_path)
        if 'DATE' in df.columns:
            df = df.rename(columns={'DATE': 'date'})
        # Register DataFrame and load into DuckDB
        con.register('prepared_df', df)
        col_str = ', '.join(df.columns)
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT {col_str} FROM prepared_df
        """)
        
        # ตรวจสอบผลลัพธ์
        result_check = con.execute(f"""
            SELECT COUNT(*) as row_count,
                   MIN(date) as min_date,
                   MAX(date) as max_date
            FROM {table_name}
        """).fetchone()
        
        result = {
            'status': 'success',
            'operation': 'direct_load',
            'source_file': prepared_parquet_path,
            'target_table': table_name,
            'loaded_rows': result_check[0],
            'data_range': f"{result_check[1]} to {result_check[2]}"
        }
        
        print(f"✅ Prepared data loaded:")
        print(f"   📊 Loaded: {result_check[0]:,} rows")
        print(f"   📅 Range: {result['data_range']}")
        print(f"   📥 Direct load from prepared parquet")
        
        return result
        
    except Exception as e:
        print(f"❌ Error loading prepared data: {e}")
        raise e
    finally:
        con.close()

# Backward compatibility wrappers
def load_raw_to_duckdb(raw_parquet_path: str, duckdb_path: str, table_name: str = "climate_raw"):
    """
    Wrapper for backward compatibility
    ตอนนี้ใช้ smart loading (auto-detect fresh vs incremental)
    """
    result = smart_load_raw_to_duckdb(raw_parquet_path, duckdb_path, table_name)
    return duckdb_path

def load_prepared_to_duckdb(prepared_parquet_path: str, duckdb_path: str, table_name: str = "climate_clean"):
    """
    Wrapper for backward compatibility  
    ตอนนี้ load จาก prepared parquet โดยตรง
    """
    print(f"📥 Loading {table_name} from prepared parquet...")
    
    result = load_prepared_to_duckdb_direct(prepared_parquet_path, duckdb_path, table_name)
    
    return duckdb_path

def load_features_to_duckdb(
    features_file_path: str,
    duckdb_path: str,
    table_name: str = "climate_features"
) -> dict:
    """
    Load features (CSV/Parquet) → DuckDB table สำหรับ feature engineering โดยเฉพาะ
    """
    import os
    import duckdb

    if not os.path.exists(features_file_path):
        raise FileNotFoundError(f"Features file not found: {features_file_path}")

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)
    con = duckdb.connect(duckdb_path)

    try:
        print(f"📥 LOADING FEATURES: {features_file_path} → {table_name}")
        if features_file_path.endswith(".csv"):
            con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{features_file_path}')
            """)
        else:
            con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_parquet('{features_file_path}')
            """)
        result_check = con.execute(f"""
            SELECT COUNT(*) as row_count,
                   MIN(date) as min_date,
                   MAX(date) as max_date
            FROM {table_name}
        """).fetchone()
        result = {
            'status': 'success',
            'operation': 'features_load',
            'source_file': features_file_path,
            'target_table': table_name,
            'loaded_rows': result_check[0],
            'data_range': f"{result_check[1]} to {result_check[2]}"
        }
        print(f"✅ Features loaded: {result_check[0]:,} rows")
        return result
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        raise e
    finally:
        con.close()