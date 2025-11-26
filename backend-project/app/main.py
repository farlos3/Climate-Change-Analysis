from fastapi import FastAPI, HTTPException, Request
from src.api_utils import get_features_json
from src.predict import predict_all_models

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
            "ml_ready_check": "/data/ready_for_ml",
            "forecast_ensemble": "/forecast/ensemble"
        },
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/features")
def features_json():
    try:
        return get_features_json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading features: {e}")
    
@app.post("/predict_all")
async def predict_all_endpoint(request: Request):
    try:
        features = await request.json()  # รับ features จาก body
        return predict_all_models(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {e}")