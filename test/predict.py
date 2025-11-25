import pandas as pd
import joblib
import json
import numpy as np

# -----------------------------
# Paths
# -----------------------------
FE_PATH = "backend-project/airflow/data/prepared/feature_engineering_t2m_output.csv"
MODEL_PATH = "backend-project/test/LightGBM_Multi_T2M_Model.joblib"
FEATURE_SELECTION_PATH = "backend-project/airflow/data/prepared/t2m_selected_features.json"
OUTPUT_PATH = "predict_result.csv"

# -----------------------------
# Load meta / selected features from JSON
# -----------------------------
with open(FEATURE_SELECTION_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# รองรับ 3 เคส:
# 1) meta = {"selected_features": [...], "target_names": [...]}
# 2) meta = ["col1", "col2", ...]
# 3) meta = {"0": "col1", "1": "col2", ...}  <-- แบบที่คุณใช้ตอนนี้
selected_features = None
target_names = None

if isinstance(meta, dict):
    if "selected_features" in meta:  # เคส 1
        selected_features = meta["selected_features"]
        target_names = meta.get("target_names", None)
    else:
        # เคส 3: dict ที่ key เป็น index ("0","1",...)
        # แปลงเป็น list ตามลำดับ index
        try:
            # sort key ตามเลข 0..n แล้วดึง value
            keys_sorted = sorted(meta.keys(), key=lambda k: int(k))
            selected_features = [meta[k] for k in keys_sorted]
        except Exception as e:
            raise ValueError(
                f"ไม่สามารถ parse JSON เป็น selected_features ได้จาก dict แบบนี้: {e}"
            )
elif isinstance(meta, list):
    # เคส 2: เป็น list ตรง ๆ
    selected_features = meta
else:
    raise ValueError("รูปแบบ JSON ไม่รองรับ: ต้องเป็น dict หรือ list")

if not selected_features:
    raise ValueError("selected_features ใน JSON ว่าง หรืออ่านไม่สำเร็จ")

print("Number of selected features:", len(selected_features))

# -----------------------------
# Load FE (inference)
# -----------------------------
df_fe = pd.read_csv(FE_PATH)

# ใช้เฉพาะ selected features
missing = [c for c in selected_features if c not in df_fe.columns]

if missing:
    raise KeyError(f"The following selected_features are missing in FE data: {missing}")

X_all = df_fe[selected_features].copy()
print("X_all shape:", X_all.shape)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(MODEL_PATH)

# -----------------------------
# Predict (ใช้แถวล่าสุด)
# -----------------------------
X_pred = X_all.iloc[[-1]]  # ให้เป็น DataFrame 1 แถว
pred = model.predict(X_pred)
pred = np.asarray(pred).flatten()  # เผื่อ multi-output → (H,)

# -----------------------------
# ชื่อ column ของ output
# -----------------------------
if isinstance(meta, dict) and "target_names" in meta:
    target_names = meta["target_names"]
    if len(target_names) != len(pred):
        raise ValueError(
            f"len(target_names) ({len(target_names)}) != len(pred) ({len(pred)})"
        )
    out_cols = target_names
else:
    # fallback ถ้า JSON ไม่มี target_names → ใช้ชื่อเดิมแบบ t2m_d{h}_forecast
    out_cols = [f"t2m_d{i+1}_forecast" for i in range(len(pred))]

# -----------------------------
# Save result
# -----------------------------
from datetime import timedelta

if "date" in df_fe.columns:
    last_date = pd.to_datetime(df_fe["date"].iloc[-1])
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, len(pred)+1)]
    # สร้างแต่ละแถวเป็น (date, t2m_d1_forecast, ...)
    rows = []
    for i in range(len(pred)):
        row = {"date": forecast_dates[i].strftime("%Y-%m-%d"), out_cols[i]: pred[i]}
        # ถ้า multi-output (เช่น 7 วัน) ให้เติม NaN ใน column อื่น
        for j in range(len(out_cols)):
            if j != i:
                row[out_cols[j]] = np.nan
        rows.append(row)
    result_df = pd.DataFrame(rows)
else:
    result_df = pd.DataFrame([pred], columns=out_cols)

result_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved prediction to {OUTPUT_PATH}")
print(result_df)