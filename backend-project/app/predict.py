import pandas as pd
import joblib
import json
import numpy as np
from datetime import timedelta

# Paths for Airflow container/production
MODEL_PATHS = {
    "lightgbm": "/opt/app/models/LightGBM_Multi_T2M_Model.joblib",
    "xgboost": "/opt/app/models/Xgboost_Multi_T2M_Model.joblib",
    "randomforest": "/opt/app/models/RandomForest_Multi_T2M_Model.joblib"
}
FEATURE_SELECTION_PATH = "/opt/airflow/data/t2m_selected_features.json"

def predict_all_models(features):
    """
    Predict 7-day T2M forecast for RF, LGBM, XGBoost, and Ensemble. Return all results as JSON.
    features: list of dicts or pandas.DataFrame
    """
    with open(FEATURE_SELECTION_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    selected_features = None
    target_names = None
    if isinstance(meta, dict):
        if "selected_features" in meta:
            selected_features = meta["selected_features"]
            target_names = meta.get("target_names", None)
        else:
            keys_sorted = sorted(meta.keys(), key=lambda k: int(k))
            selected_features = [meta[k] for k in keys_sorted]
    elif isinstance(meta, list):
        selected_features = meta
    if not selected_features:
        raise Exception("selected_features not found")

    # Convert features to DataFrame
    if isinstance(features, pd.DataFrame):
        df = features
    elif isinstance(features, list):
        df = pd.DataFrame(features)
    elif isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        raise Exception("features must be dict, list of dicts, or DataFrame")

    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise Exception(f"Missing features: {missing}")
    X_pred = df[selected_features].iloc[[-1]]

    rf_model = joblib.load(MODEL_PATHS["randomforest"])
    lgbm_model = joblib.load(MODEL_PATHS["lightgbm"])
    xgb_model = joblib.load(MODEL_PATHS["xgboost"])

    preds_rf = rf_model.predict(X_pred).flatten()
    preds_lgbm = lgbm_model.predict(X_pred).flatten()
    preds_xgb = xgb_model.predict(X_pred).flatten()
    preds_ensemble = np.mean([preds_rf, preds_lgbm, preds_xgb], axis=0)

    if target_names and len(target_names) == len(preds_rf):
        out_cols = target_names
    else:
        out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_rf))]

    def build_row(preds, model_name):
        return [{out_cols[i]: float(preds[i]), "model": model_name} for i in range(len(preds))]
    results = {
        "randomforest": build_row(preds_rf, "randomforest"),
        "lightgbm": build_row(preds_lgbm, "lightgbm"),
        "xgboost": build_row(preds_xgb, "xgboost"),
        "ensemble": build_row(preds_ensemble, "ensemble")
    }
    return results