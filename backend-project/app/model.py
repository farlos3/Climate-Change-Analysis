import duckdb
import pandas as pd
import joblib
import json
import numpy as np
from datetime import timedelta

DUCKDB_PATH = "/opt/airflow/data/duckdb/climate.duckdb"
FEATURE_TABLE = "climate_features"
FORECAST_TABLE = "climate_forecast"
MODEL_PATHS = {
	"lightgbm": "/opt/app/models/LightGBM_Multi_T2M_Model.joblib",
	"xgboost": "/opt/app/models/Xgboost_Multi_T2M_Model.joblib",
	"randomforest": "/opt/app/models/RandomForest_Multi_T2M_Model.joblib"
}

FEATURE_SELECTION_PATH = "/opt/app/data/prepared/t2m_selected_features.json"

def predict_ensemble():
	"""
	Predict 7-day T2M forecast using ensemble model, return as list of dicts and save to DuckDB
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

	con = duckdb.connect(DUCKDB_PATH)
	df_fe = con.execute(f"SELECT * FROM {FEATURE_TABLE}").df()
	con.close()
	missing = [c for c in selected_features if c not in df_fe.columns]
	if missing:
		raise Exception(f"Missing features: {missing}")
	X_pred = df_fe[selected_features].iloc[[-1]]

	rf_model = joblib.load(MODEL_PATHS["randomforest"])
	lgbm_model = joblib.load(MODEL_PATHS["lightgbm"])
	xgb_model = joblib.load(MODEL_PATHS["xgboost"])
	def ensemble_predict(X):
		preds_rf = rf_model.predict(X)
		preds_lgbm = lgbm_model.predict(X)
		preds_xgb = xgb_model.predict(X)
		return np.mean([preds_rf, preds_lgbm, preds_xgb], axis=0)

	pred = ensemble_predict(X_pred)
	pred = np.asarray(pred).flatten()

	if target_names and len(target_names) == len(pred):
		out_cols = target_names
	else:
		out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(pred))]

	if "date" in df_fe.columns:
		last_date = pd.to_datetime(df_fe["date"].iloc[-1])
		forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(pred)+1)]
		rows = []
		for i in range(len(pred)):
			row = {
				"date_predicted": last_date.strftime("%Y-%m-%d"),
				"date_target": forecast_dates[i].strftime("%Y-%m-%d"),
				"horizon": i+1,
				"prediction": float(pred[i]),
				"model": "ensemble"
			}
			rows.append(row)
		result_df = pd.DataFrame(rows)
	else:
		result_df = pd.DataFrame([pred], columns=out_cols)

	# ไม่ต้องบันทึกลง DuckDB ในฟังก์ชันนี้ (API จะคืนผลลัพธ์อย่างเดียว)
	return result_df.to_dict(orient="records")
