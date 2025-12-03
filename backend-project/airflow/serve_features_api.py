from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import duckdb
import pandas as pd
import os

app = FastAPI()

DUCKDB_PATH = '/opt/airflow/data/duckdb/climate.duckdb'
FEATURE_TABLE = "feature_store"

@app.get('/features')
def features_json():
    con = duckdb.connect("md:Climate Change (T2M)") 
    df = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
    con.close()
    return JSONResponse(content=df.to_dict(orient="records"))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8090)