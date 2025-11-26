from fastapi import FastAPI, HTTPException, Request, Body
from predict import predict_all_models, predict_randomforest, predict_lightgbm, predict_xgboost, predict_ensemble

app = FastAPI(title="Climate Data Backend")

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Climate Data Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
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

# เพิ่ม endpoint สำหรับแต่ละโมเดล
@app.get("/predict/randomforest")
def predict_rf():
    if not latest_features or not latest_features.get("features"):
        raise HTTPException(status_code=404, detail="No features available for prediction")
    features = latest_features["features"]
    try:
        preds_rf, target_names = predict_randomforest(features)
        if target_names and len(target_names) == len(preds_rf):
            out_cols = target_names
        else:
            out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_rf))]
        result = [{out_cols[i]: float(preds_rf[i]), "model": "randomforest"} for i in range(len(preds_rf))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": result}

@app.get("/predict/lightgbm")
def predict_lgbm():
    if not latest_features or not latest_features.get("features"):
        raise HTTPException(status_code=404, detail="No features available for prediction")
    features = latest_features["features"]
    try:
        preds_lgbm, target_names = predict_lightgbm(features)
        if target_names and len(target_names) == len(preds_lgbm):
            out_cols = target_names
        else:
            out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_lgbm))]
        result = [{out_cols[i]: float(preds_lgbm[i]), "model": "lightgbm"} for i in range(len(preds_lgbm))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": result}

@app.get("/predict/xgboost")
def predict_xgb():
    if not latest_features or not latest_features.get("features"):
        raise HTTPException(status_code=404, detail="No features available for prediction")
    features = latest_features["features"]
    try:
        preds_xgb, target_names = predict_xgboost(features)
        if target_names and len(target_names) == len(preds_xgb):
            out_cols = target_names
        else:
            out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_xgb))]
        result = [{out_cols[i]: float(preds_xgb[i]), "model": "xgboost"} for i in range(len(preds_xgb))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": result}

@app.get("/predict/ensemble")
def predict_ensemble_api():
    if not latest_features or not latest_features.get("features"):
        raise HTTPException(status_code=404, detail="No features available for prediction")
    features = latest_features["features"]
    try:
        preds_ensemble, target_names = predict_ensemble(features)
        if target_names and len(target_names) == len(preds_ensemble):
            out_cols = target_names
        else:
            out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_ensemble))]
        result = [{out_cols[i]: float(preds_ensemble[i]), "model": "ensemble"} for i in range(len(preds_ensemble))]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": result}
