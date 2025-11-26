import duckdb
import pandas as pd

DUCKDB_PATH = "/opt/airflow/data/duckdb/climate.duckdb"
FEATURE_TABLE = "climate_features"

def get_features_json():
    
    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    con.close()
    return df.to_dict(orient="records")
