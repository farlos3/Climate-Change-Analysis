import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys

# เพิ่ม Path เพื่อให้สามารถ import features ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import features as f 
except ImportError:
    print("ERROR: Could not import 'features.py'. Ensure it's in the 'scripts' folder.")
    sys.exit(1)


# การกำหนดค่าคงที่และโฟลเดอร์ 
DATA_PATH = 'Dataset/clean_nasa_power_data.csv'
MODEL_DIR = 'models/xgboost'
MODEL_FILENAME = 'xgboost_tsurf_v1.json'


# ฟังก์ชันการโหลดข้อมูลและเตรียมชุด X/y
def load_and_prepare_data(path: str) -> pd.DataFrame:
    """Load cleaned data and ensure 'date' is in datetime format."""
    print(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}. Please check your 'Dataset/' folder.")
        sys.exit(1)
    except KeyError:
        print("ERROR: KeyError: 'date'. The CSV file does not contain a column named 'date'.")
        print("Please check your data preparation notebook to ensure the date column is created and named 'date'.")
        sys.exit(1)

# ฟังก์ชันการแบ่งข้อมูลตามเวลา
def temporal_split(df: pd.DataFrame, split_date: str):
    """Splits data into training and testing sets based on time."""
    split_date = pd.to_datetime(split_date)
    y = df[f.TARGET_COL]
    X = df.drop(columns=[f.TARGET_COL] + f.COLS_TO_DROP_PRE_TRAIN, errors='ignore')
    
    X_train = X[X['date'] < split_date]
    X_test = X[X['date'] >= split_date]
    y_train = y[X['date'] < split_date]
    y_test = y[X['date'] >= split_date]

    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    
    return X_train, X_test, y_train, y_test

# ฟังก์ชันการฝึกและประเมินโมเดล 
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Initializes, trains, and evaluates the XGBoost model using error handling for legacy versions."""
    
    xgb_model = xgb.XGBRegressor(**f.XGB_PARAMS)
    best_iter = f.XGB_PARAMS['n_estimators']

    print(f"Train set size: {len(X_train)} | Test set size: {len(X_test)}")
    print(f"Start training XGBoost model for {f.TARGET_COL}...")
    
    try:
        # ลองใช้ไวยากรณ์ที่ควรจะทำงานได้ (legacy syntax)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50, 
            eval_metric="rmse", 
            verbose=False
        )
        # ถ้าสำเร็จ, ดึงค่า best_iteration 
        best_iter = xgb_model.best_iteration
    except TypeError as e:
        # หากเกิด TypeError (เช่น 'early_stopping_rounds' ไม่รู้จัก)
        print(f"\nWarning: Failed to use early_stopping_rounds due to Python/XGBoost compatibility issue: {e}")
        print("Training model with full estimators (n_estimators=1000) instead.")
        # ฝึกแบบไม่มี Early Stopping
        xgb_model.fit(X_train, y_train, verbose=False)
        # best_iter ยังคงเป็นค่าเริ่มต้น (1000)
    # --------------------------------------------------------------------------
    
    y_pred = xgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"Model Training Summary ({f.TARGET_COL})")
    print(f"Best Iteration: {best_iter}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R-squared (R2 Score): {r2:.4f}")
    print("=" * 50)
    
    return xgb_model

# ฟังก์ชันหลักสำหรับรัน Pipeline 
def main():
    """Main execution pipeline."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    df = load_and_prepare_data(DATA_PATH)
    
    X_train, X_test, y_train, y_test = temporal_split(df, f.SPLIT_DATE)
    
    model = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    model.save_model(model_path) 
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    main()