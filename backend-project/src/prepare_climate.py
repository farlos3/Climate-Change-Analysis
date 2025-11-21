import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


# Missing value flags used by NASA POWER
MISSING_FLAGS = [-99, -999, -9999, -8888, -7777]


def prepare_climate_data(
    duckdb_path: str,
    prepared_parquet_path: str,
    scaler_path: str = "/opt/airflow/models/feature_scaler.pkl",
    raw_table: str = "climate_raw"
):

    if not os.path.exists(duckdb_path):
        raise FileNotFoundError(f"DuckDB not found: {duckdb_path}")

    con = duckdb.connect(duckdb_path)
    df = con.execute(f"SELECT * FROM {raw_table};").df()
    con.close()

    # ถ้ามี index column ให้เปลี่ยนเป็น date
    if '__index_level_0__' in df.columns:
        df['date'] = pd.to_datetime(df['__index_level_0__'])
        df = df.drop(columns=['__index_level_0__'])
    elif 'date' not in df.columns:
        # ถ้าไม่มี date column หา column ที่เป็น datetime
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            df['date'] = df[date_cols[0]]
        else:
            raise ValueError("No date column found in the raw data")
    
    df = df.sort_values('date')

    df.columns = df.columns.str.lower()

    nan_cols = df.columns[df.isna().any()].tolist()
    df = df.drop(columns=nan_cols)

    bad_cols = [
        'airmass', 'allsky_kt', 'allsky_nkt', 'allsky_sfc_lw_dwn', 'allsky_sfc_lw_up',
        'allsky_sfc_par_diff', 'allsky_sfc_par_dirh', 'allsky_sfc_par_tot',
        'allsky_sfc_sw_diff', 'allsky_sfc_sw_dirh', 'allsky_sfc_sw_dni',
        'allsky_sfc_sw_dwn', 'allsky_sfc_sw_up', 'allsky_sfc_uva', 'allsky_sfc_uvb',
        'allsky_sfc_uv_index', 'allsky_srf_alb', 'aod_55', 'aod_55_adj', 'aod_84',
        'gwm_height', 'gwm_height_anomaly', 'imerg_precliquid_prob', 'imerg_prectot',
        'imerg_prectot_count', 'midday_insol', 'original_allsky_sfc_lw_dwn',
        'original_allsky_sfc_sw_diff', 'original_allsky_sfc_sw_dirh',
        'original_allsky_sfc_sw_dwn', 'original_clrsky_sfc_lw_dwn',
        'original_clrsky_sfc_sw_dwn', 'psh', 'pw', 'srf_alb_adj', 'sza',
        'toa_sw_dni', 'toa_sw_dwn', 'ts_adj'
    ]

    df = df.drop(columns=bad_cols, errors='ignore')

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    # Remove last 3 (data delay problem)
    last3 = df['date'].nlargest(3)
    df = df[~df['date'].isin(last3)]

    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

    for col in numeric_cols:
        mask = df[col].isin(MISSING_FLAGS)
        if mask.any():
            df.loc[mask, col] = df.loc[~mask, col].mean()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.dayofweek

    # Asia seasons
    df['season_num'] = df['month'].apply(
        lambda m: 0 if m in [12,1,2] else 1 if m in [3,4,5] else 2
    )

    # Sometimes NASA includes these but they are mostly zero
    df = df.drop(columns=['frost_days', 'precsno', 'precsnoland',
                          'snodp', 'frsno', 'frseaice'], errors='ignore')

    H = 7
    df['et_total'] = df['evland'] + df['evptrns']

    df["t2m_forecast_7d"] = df["t2m"].shift(-H)
    df["rain_forecast_7d"] = df["prectotcorr"].shift(-H)
    df["et_forecast_7d"] = df["et_total"].shift(-H)
    df["soil_moisture_forecast_7d"] = df["gwettop"].shift(-H)
    df["wind_forecast_7d"] = df["ws10m"].shift(-H)

    target_cols = [
        "t2m_forecast_7d",
        "rain_forecast_7d",
        "et_forecast_7d",
        "soil_moisture_forecast_7d",
        "wind_forecast_7d",
    ]

    df = df.dropna(subset=target_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in target_cols]

    # Load existing scaler or train new
    if os.path.exists(scaler_path):
        print(f"Loading existing scaler → {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        print("No scaler found → Training new scaler...")
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Saved new scaler: {scaler_path}")

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    os.makedirs(os.path.dirname(prepared_parquet_path), exist_ok=True)
    df_scaled.to_parquet(prepared_parquet_path, index=False)

    print(f"Saved CLEAN parquet: {prepared_parquet_path}")

    return prepared_parquet_path
