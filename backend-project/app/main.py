from fastapi import FastAPI, HTTPException, Request, Body
from predict import predict_all_models

app = FastAPI(title="Climate Data Backend")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Climate Data Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "all_data": "/data/all", 
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }
    
latest_features = {}

@app.post("/ingest/features")
def ingest_features(payload: dict = Body(...)):
    global latest_features
    features = payload.get("features")
    row_count = payload.get("row_count")
    latest_features = {
        "features": features,
        "row_count": row_count
    }

    # print(f"Sample row: {features[0]}")
    print(f"Received features from Airflow, rows={row_count}")
    
    return {"status": "received", "rows": row_count}

# เพิ่ม endpoint สำหรับ predict จาก features ล่าสุด
@app.get("/predict/latest")
def predict_latest():
    from predict import predict_all_models
    if not latest_features or not latest_features.get("features"):
        raise HTTPException(status_code=404, detail="No features available for prediction")
    features = latest_features["features"]
    try:
        result = predict_all_models(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": result}

@app.get("/features/latest")
def get_latest_features():
    if not latest_features:
        raise HTTPException(status_code=404, detail="No features available")
    return latest_features

@app.get("/ingest/features")
def ingest_features_get():
    return {"message": "Use POST to ingest features from Airflow."}