import os
import duckdb

def load_raw_to_duckdb(raw_parquet_path: str, duckdb_path: str, table_name: str = "climate_raw"):
    """Load raw parquet data into DuckDB"""
    
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Raw parquet not found: {raw_parquet_path}")

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

    con = duckdb.connect(duckdb_path)

    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_parquet('{raw_parquet_path}');
        """)

        print(f"Loaded raw data into DuckDB: {duckdb_path} (table: {table_name})")

    except Exception as e:
        print(f"Error loading raw data to DuckDB: {e}")
        raise e

    finally:
        con.close()

    return duckdb_path

def load_prepared_to_duckdb(prepared_parquet_path: str, duckdb_path: str, table_name: str = "climate_clean"):
    """Load prepared/clean parquet data into DuckDB"""
    
    if not os.path.exists(prepared_parquet_path):
        raise FileNotFoundError(f"Prepared parquet not found: {prepared_parquet_path}")

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

    con = duckdb.connect(duckdb_path)

    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_parquet('{prepared_parquet_path}');
        """)

        # Optional: Optimize table metadata
        con.execute("VACUUM;")

        print(f"Loaded prepared data into DuckDB: {duckdb_path} (table: {table_name})")

    except Exception as e:
        print(f"Error loading prepared data to DuckDB: {e}")
        raise e

    finally:
        con.close()

    return duckdb_path

def load_to_duckdb(prepared_parquet_path: str, duckdb_path: str, table_name: str = "climate_clean"):

    if not os.path.exists(prepared_parquet_path):
        raise FileNotFoundError(f"Prepared parquet not found: {prepared_parquet_path}")

    os.makedirs(os.path.dirname(duckdb_path), exist_ok=True)

    con = duckdb.connect(duckdb_path)

    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_parquet('{prepared_parquet_path}');
        """)

        # Optional: Optimize table metadata
        con.execute("VACUUM;")

        print(f"Loaded data into DuckDB: {duckdb_path} (table: {table_name})")

    except Exception as e:
        print(f"Error loading to DuckDB: {e}")
        raise e

    finally:
        con.close()

    return duckdb_path