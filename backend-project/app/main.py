from fastapi import FastAPI, HTTPException
import duckdb
import os
import pandas as pd

app = FastAPI(title="Climate Data Backend")

# เส้นทางไฟล์ DuckDB (volume ถูก mount เข้ามาใน /app/data/duckdb/climate.duckdb)
DB_PATH = os.getenv("DUCKDB_PATH", "/app/data/duckdb/climate.duckdb")

def get_connection(read_only=True):
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DuckDB file not found at: {DB_PATH}")
    return duckdb.connect(DB_PATH, read_only=read_only)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Climate Data Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "all_data": "/data/all", 
            "ml_ready_check": "/data/ready_for_ml"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
def health():
    return {"status": "ok", "duckdb_found": os.path.exists(DB_PATH)}

@app.get("/data/all")
def data_all():
    try:
        con = get_connection()
        df = con.execute("SELECT * FROM climate_clean;").df()
        con.close()

        return {
            "row_count": len(df),
            "rows": df.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/ready_for_ml")
def ready_for_ml():
    try:
        con = get_connection()

        row_count = con.execute(
            "SELECT COUNT(*) FROM climate_clean;"
        ).fetchone()[0]

        # เช็ค null ของคอลัมน์สำคัญ
        null_stats = con.execute("""
            SELECT
                SUM(CASE WHEN t2m IS NULL THEN 1 ELSE 0 END) AS t2m_nulls,
                SUM(CASE WHEN rainfall IS NULL THEN 1 ELSE 0 END) AS rainfall_nulls,
                SUM(CASE WHEN rh2m IS NULL THEN 1 ELSE 0 END) AS rh2m_nulls,
                SUM(CASE WHEN ws10m IS NULL THEN 1 ELSE 0 END) AS ws10m_nulls
            FROM climate_clean
        """).df().to_dict(orient="records")[0]

        con.close()

        # Dataset พร้อมถ้า: มีข้อมูล + ไม่มี null
        ready = row_count > 0 and all(v == 0 for v in null_stats.values())

        return {
            "row_count": row_count,
            "null_stats": null_stats,
            "ready_for_ml": ready
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))