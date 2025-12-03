import pandas as pd
import joblib
import json
import numpy as np
from datetime import timedelta

# Paths for Airflow container/production
MODEL_PATHS = {
    "lightgbm": "/opt/airflow/data/models/LightGBM_Multi_T2M_Model.joblib",
    "xgboost": "/opt/airflow/data/models/Xgboost_Multi_T2M_Model.joblib",
    "randomforest": "/opt/airflow/data/models/RandomForest_Multi_T2M_Model.joblib"
}
FEATURE_SELECTION_PATH = "/opt/airflow/data/t2m_selected_features.json"

def load_selected_features():
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
    return selected_features, target_names

def prepare_features(features, selected_features):
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
    return df[selected_features].iloc[[-1]]

def predict_randomforest(features):
    selected_features, target_names = load_selected_features()
    X_pred = prepare_features(features, selected_features)
    rf_model = joblib.load(MODEL_PATHS["randomforest"])
    preds_rf = rf_model.predict(X_pred).flatten()
    return preds_rf, target_names

def predict_all_models(features):
    preds_rf, target_names = predict_randomforest(features)
    preds_lgbm, _ = predict_lightgbm(features)
    preds_xgb, _ = predict_xgboost(features)
    preds_ensemble, _ = predict_ensemble(features)
    if target_names and len(target_names) == len(preds_rf):
        out_cols = target_names
    else:
        out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_rf))]
    results = {
        "randomforest": build_row(preds_rf, "randomforest", out_cols),
        "lightgbm": build_row(preds_lgbm, "lightgbm", out_cols),
        "xgboost": build_row(preds_xgb, "xgboost", out_cols),
        "ensemble": build_row(preds_ensemble, "ensemble", out_cols)
    }
    return results

def predict_lightgbm(features):
    selected_features, target_names = load_selected_features()
    X_pred = prepare_features(features, selected_features)
    lgbm_model = joblib.load(MODEL_PATHS["lightgbm"])
    preds_lgbm = lgbm_model.predict(X_pred).flatten()
    return preds_lgbm, target_names

def predict_xgboost(features):
    selected_features, target_names = load_selected_features()
    X_pred = prepare_features(features, selected_features)
    xgb_model = joblib.load(MODEL_PATHS["xgboost"])
    preds_xgb = xgb_model.predict(X_pred).flatten()
    return preds_xgb, target_names

def predict_ensemble(features):
    preds_rf, target_names = predict_randomforest(features)
    preds_lgbm, _ = predict_lightgbm(features)
    preds_xgb, _ = predict_xgboost(features)
    preds_ensemble = np.mean([preds_rf, preds_lgbm, preds_xgb], axis=0)
    return preds_ensemble, target_names

def build_row(preds, model_name, out_cols):
    return [{out_cols[i]: float(preds[i]), "model": model_name} for i in range(len(preds))]

def predict_all_models(features):
    preds_rf, target_names = predict_randomforest(features)
    preds_lgbm, _ = predict_lightgbm(features)
    preds_xgb, _ = predict_xgboost(features)
    preds_ensemble, _ = predict_ensemble(features)
    if target_names and len(target_names) == len(preds_rf):
        out_cols = target_names
    else:
        out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(preds_rf))]
    results = {
        "randomforest": build_row(preds_rf, "randomforest", out_cols),
        "lightgbm": build_row(preds_lgbm, "lightgbm", out_cols),
        "xgboost": build_row(preds_xgb, "xgboost", out_cols),
        "ensemble": build_row(preds_ensemble, "ensemble", out_cols)
    }
    return results