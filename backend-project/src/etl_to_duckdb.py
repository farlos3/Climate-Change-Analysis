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
        
        if not table_exists:
            # Fresh Start: สร้าง table ใหม่
            print(f"🆕 FRESH START: Creating new {table_name} table...")
            
            con.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_parquet('{raw_parquet_path}')
            """)
            
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            date_range = con.execute(f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name}").fetchone()
            
            result = {
                'status': 'fresh_start',
                'operation': 'create_new_table',
                'total_rows': row_count,
                'date_range': f"{date_range[0]} to {date_range[1]}",
                'table_name': table_name
            }
            
            print(f"✅ Fresh start completed: {row_count:,} rows loaded")
            
        else:
            # Incremental Update: ทับข้อมูลที่ overlap
            print(f"🔄 INCREMENTAL UPDATE: Updating {table_name} table...")
            
            # อ่านข้อมูลใหม่จาก parquet
            new_data = con.execute(f"SELECT * FROM read_parquet('{raw_parquet_path}')").df()
            
            if new_data.empty:
                return {'status': 'no_data', 'message': 'No new data to process'}
            
            # หาช่วงวันที่ของข้อมูลใหม่
            date_range = {
                'start_date': new_data['date'].min(),
                'end_date': new_data['date'].max(),
                'total_new_rows': len(new_data)
            }
            
            # ตรวจสอบข้อมูลที่จะ overlap
            overlap_check = con.execute(f"""
                SELECT COUNT(*) as overlap_count
                FROM {table_name}
                WHERE date >= '{date_range['start_date']}' 
                AND date <= '{date_range['end_date']}'
            """).fetchone()[0]
            
            # ลบข้อมูลเก่าที่ overlap
            if overlap_check > 0:
                print(f"   🗑️  Removing {overlap_check:,} overlapping rows...")
                con.execute(f"""
                    DELETE FROM {table_name}
                    WHERE date >= '{date_range['start_date']}' 
                    AND date <= '{date_range['end_date']}'
                """)
            
            # Insert ข้อมูลใหม่
            print(f"   ➕ Inserting {date_range['total_new_rows']:,} new rows...")
            con.execute(f"""
                INSERT INTO {table_name}
                SELECT * FROM read_parquet('{raw_parquet_path}')
            """)
            
            # ตรวจสอบผลลัพธ์สุดท้าย
            final_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            final_range = con.execute(f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name}").fetchone()
            
            result = {
                'status': 'incremental_update',
                'operation': 'overlap_and_insert',
                'new_data_range': f"{date_range['start_date']} to {date_range['end_date']}",
                'new_rows_added': date_range['total_new_rows'],
                'overlap_rows_replaced': overlap_check,
                'total_rows_after': final_count,
                'full_data_range': f"{final_range[0]} to {final_range[1]}",
                'table_name': table_name
            }
            
            print(f"✅ Incremental update completed: {final_count:,} total rows")
            print(f"   📊 Added: {date_range['total_new_rows']} | Replaced: {overlap_check}")
        
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
        
        # Simple load: อ่าน parquet เข้า DuckDB ตรงๆ
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_parquet('{prepared_parquet_path}')
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