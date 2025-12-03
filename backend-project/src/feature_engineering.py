import pandas as pd
import numpy as np
from typing import List, Tuple
import duckdb

SELECTED_FEATURES = [
    # Soil Temperature
    "tsoil1", "tsoil2", "tsoil3", "tsoil4",

    # Surface / Air Temperature
    "t2m", "t2m_max", "t2m_min",
    "t10m", "t10m_max", "t10m_min",
    "ts", "ts_min", "ts_max", "tsurf",
    "t2m_range", "t10m_range", "ts_range",

    # Wind (V component)
    "v2m", "v10m", "v50m",

    # Humidity / Air Density
    "rhoa", "t2mwet",

    # Soil Moisture
    "gwettop", "gwetroot", "gwetprof",

    # ET / Evapotranspiration
    "evptrns", "et_total", "evland",

    # Physical properties
    "z0m", "to3",

    # Seasonal (for sin/cos)
    "month",
]

ROLLING_COLS = [
    "t2m", "t2m_max", "t2m_min",
    "t10m", "t10m_max", "t10m_min",
    "ts", "ts_max", "ts_min", "tsurf",
    "tsoil1", "tsoil2", "tsoil3", "tsoil4",
    "t2mwet", "rhoa",
    "gwettop", "gwetroot", "gwetprof",
    "v2m", "v10m", "v50m"
]

ROLLING_WINDOWS = [3, 7]
LAG_LIST = list(range(1, 8))  # lag 1–7 วัน


# Base FE functions
def load_and_sort_data(input_path: str) -> pd.DataFrame:
    """
    โหลดข้อมูลจาก Parquet และ sort ตามวันที่ (date)
    """
    df = pd.read_parquet(input_path)
    df = df.sort_values("date")
    return df


def select_base_features(df: pd.DataFrame, selected_features: List[str]) -> List[str]:
    """
    เลือก base features ที่ยังมีอยู่จริงใน DataFrame
    """
    base_features = [c for c in selected_features if c in df.columns]
    return base_features


def build_base_frame(df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
    """
    สร้าง df_sel = [date] + base_features
    (ไม่ดึง target ใด ๆ มาใช้เลย)
    """
    cols = ["date"] + base_features
    df_sel = df[cols].copy()
    return df_sel


def add_seasonal_features(df_sel: pd.DataFrame) -> pd.DataFrame:
    """
    สร้าง seasonal features (month_sin, month_cos, doy_sin, doy_cos)
    จาก date และ month ตามที่ใช้ตอน train
    """
    df_sel["date"] = pd.to_datetime(df_sel["date"])
    df_sel["dayofyear"] = df_sel["date"].dt.dayofyear

    seasonal_df = pd.DataFrame({
        "month_sin": np.sin(2 * np.pi * df_sel["month"] / 12),
        "month_cos": np.cos(2 * np.pi * df_sel["month"] / 12),
        "doy_sin": np.sin(2 * np.pi * df_sel["dayofyear"] / 365),
        "doy_cos": np.cos(2 * np.pi * df_sel["dayofyear"] / 365),
    }, index=df_sel.index)

    return seasonal_df


def add_lag_features(
    df_sel: pd.DataFrame,
    base_features: List[str],
    lags: List[int] = LAG_LIST
) -> pd.DataFrame:
    """
    สร้าง lag feature ย้อนหลังตาม list ของ lags
    ข้ามคอลัมน์ 'month' เหมือนโน้ตบุ๊ก
    """
    lag_frames = []

    for col in base_features:
        if col == "month":
            continue
        for lag in lags:
            lag_series = df_sel[col].shift(lag).rename(f"{col}_lag{lag}")
            lag_frames.append(lag_series)

    if lag_frames:
        lag_df = pd.concat(lag_frames, axis=1)
    else:
        lag_df = pd.DataFrame(index=df_sel.index)

    return lag_df


def add_rolling_features(
    df_sel: pd.DataFrame,
    rolling_cols: List[str] = ROLLING_COLS,
    windows: List[int] = ROLLING_WINDOWS
) -> pd.DataFrame:
    """
    สร้าง rolling window features (mean, std) สำหรับคอลัมน์ที่กำหนด
    และหน้าต่างเวลาใน windows
    """
    rolling_frames = []

    for col in rolling_cols:
        if col not in df_sel.columns:
            # ถ้าคอลัมน์หายไปจาก df_sel ก็ข้ามไป
            continue
        for w in windows:
            rolling_mean = df_sel[col].rolling(w).mean().rename(f"{col}_roll{w}_mean")
            rolling_std = df_sel[col].rolling(w).std().rename(f"{col}_roll{w}_std")
            rolling_frames.extend([rolling_mean, rolling_std])

    if rolling_frames:
        rolling_df = pd.concat(rolling_frames, axis=1)
    else:
        rolling_df = pd.DataFrame(index=df_sel.index)

    return rolling_df

# Advanced FE function
def add_advanced_features(df_fe: pd.DataFrame) -> pd.DataFrame:
    """
    เพิ่ม advanced features ตาม notebook:
      - Volatility
      - Weather pattern
      - Interactions
      - Statistical
      - Temporal
      - Advanced lags for derived vars
    ใช้ df_fe ที่มี:
      - t2m, t10m, ts, tsurf, tsoil1-4, v10m, rh
      - month, dayofyear ฯลฯ
    """
    print("Adding advanced features...")

    # 1. Temperature Volatility Features
    volatility_df = pd.DataFrame({
        # Temperature volatility (coefficient of variation)
        "t2m_volatility_3d": df_fe["t2m"].rolling(3).std() / df_fe["t2m"].rolling(3).mean(),
        "t2m_volatility_7d": df_fe["t2m"].rolling(7).std() / df_fe["t2m"].rolling(7).mean(),

        # Daily changes and acceleration
        "t2m_daily_change": df_fe["t2m"].diff(),
        "t2m_acceleration": df_fe["t2m"].diff().diff(),

        # Temperature volatility for other temp variables
        "t10m_volatility_3d": df_fe["t10m"].rolling(3).std() / df_fe["t10m"].rolling(3).mean(),
        "ts_volatility_3d": df_fe["ts"].rolling(3).std() / df_fe["ts"].rolling(3).mean(),
        "tsurf_volatility_3d": df_fe["tsurf"].rolling(3).std() / df_fe["tsurf"].rolling(3).mean(),
    }, index=df_fe.index)
    print(f"   - Added {len(volatility_df.columns)} volatility features")

    # 2. Weather Pattern Features
    weather_df = pd.DataFrame({
        # Heating/Cooling degree days (comfort ~26°C)
        "hdd": np.maximum(26 - df_fe["t2m"], 0),
        "cdd": np.maximum(df_fe["t2m"] - 26, 0),

        # Temperature anomalies
        "t2m_monthly_avg": df_fe.groupby("month")["t2m"].transform("mean"),
        "t2m_anomaly": df_fe["t2m"] - df_fe.groupby("month")["t2m"].transform("mean"),

        # Seasonal temperature difference
        "t2m_seasonal_trend": df_fe.groupby("month")["t2m"].transform("std"),
        "t2m_vs_seasonal_avg": df_fe["t2m"] - df_fe.groupby("month")["t2m"].transform("mean"),

        # Extreme temperature indicators
        "t2m_is_extreme_hot": (df_fe["t2m"] > df_fe["t2m"].quantile(0.95)).astype(int),
        "t2m_is_extreme_cold": (df_fe["t2m"] < df_fe["t2m"].quantile(0.05)).astype(int),
    }, index=df_fe.index)
    print(f"   - Added {len(weather_df.columns)} weather pattern features")

    # 3. Cross-variable Interactions
    interaction_df = pd.DataFrame({
        # Temperature gradients
        "temp_gradient_surface": df_fe["t2m"] - df_fe["ts"],
        "temp_gradient_soil": df_fe["tsoil1"] - df_fe["tsoil4"],
        "temp_gradient_altitude": df_fe["t10m"] - df_fe["t2m"],

        # Soil temperature gradients
        "soil_temp_gradient_shallow": df_fe["tsoil1"] - df_fe["tsoil2"],
        "soil_temp_gradient_deep": df_fe["tsoil2"] - df_fe["tsoil4"],

        # Wind-temperature interactions
        "wind_chill_factor": df_fe["t2m"] - (df_fe["v10m"] * 0.1),
        "heat_index_simple": df_fe["t2m"] + (df_fe["rhoa"] * 0.01),

        # Moisture-temperature interactions
        "temp_humidity_ratio": df_fe["t2m"] / (df_fe["rhoa"] + 1),
        "evaporation_potential": df_fe["t2m"] * df_fe["v10m"] * 0.01,

        # Soil moisture vs temperature
        "soil_temp_moisture": df_fe["tsoil1"] * df_fe["gwettop"],
        "surface_efficiency": df_fe["ts"] / (df_fe["evptrns"] + 0.1),
    }, index=df_fe.index)
    print(f"   - Added {len(interaction_df.columns)} interaction features")

    # 4. Statistical Pattern Features
    statistical_df = pd.DataFrame({
        # Moving percentiles
        "t2m_rolling_p25": df_fe["t2m"].rolling(7).quantile(0.25),
        "t2m_rolling_p75": df_fe["t2m"].rolling(7).quantile(0.75),
        "t2m_rolling_iqr": df_fe["t2m"].rolling(7).quantile(0.75)
        - df_fe["t2m"].rolling(7).quantile(0.25),

        # Relative position in recent window
        "t2m_position_in_week": df_fe["t2m"] / (df_fe["t2m"].rolling(7).max() + 0.001),
        "t2m_relative_to_recent_max": df_fe["t2m"] - df_fe["t2m"].rolling(7).max(),
        "t2m_relative_to_recent_min": df_fe["t2m"] - df_fe["t2m"].rolling(7).min(),

        # Momentum indicators
        "t2m_momentum_3d": df_fe["t2m"] - df_fe["t2m"].shift(3),
        "t2m_momentum_7d": df_fe["t2m"] - df_fe["t2m"].shift(7),

        # Trend strength
        "t2m_trend_strength": (
            df_fe["t2m"].rolling(7).mean() - df_fe["t2m"].rolling(14).mean()
        ),
    }, index=df_fe.index)
    print(f"   - Added {len(statistical_df.columns)} statistical features")

    # 5. Cyclical and Temporal Features
    temporal_df = pd.DataFrame({
        # Multi-frequency seasonal patterns
        "week_sin": np.sin(2 * np.pi * df_fe["dayofyear"] / 7),
        "week_cos": np.cos(2 * np.pi * df_fe["dayofyear"] / 7),

        # Quarterly patterns
        "quarter_sin": np.sin(2 * np.pi * (df_fe["month"] - 1) / 3),
        "quarter_cos": np.cos(2 * np.pi * (df_fe["month"] - 1) / 3),

        # Half-year patterns
        "half_year_sin": np.sin(2 * np.pi * (df_fe["month"] - 1) / 6),
        "half_year_cos": np.cos(2 * np.pi * (df_fe["month"] - 1) / 6),

        # Day position in month (approx)
        "month_progress": (df_fe["dayofyear"] % 30) / 30,
    }, index=df_fe.index)
    print(f"   - Added {len(temporal_df.columns)} temporal features")

    # 6. Lag Features for New Variables
    new_vars_for_lag = [
        "t2m_daily_change",
        "t2m_volatility_3d",
        "t2m_anomaly",
        "temp_gradient_surface",
        "wind_chill_factor",
    ]

    advanced_lag_frames = []
    for var in new_vars_for_lag:
        for lag in [1, 2, 3]:  # Short-term lags for derived features
            if var in volatility_df.columns:
                series = volatility_df[var]
            elif var in weather_df.columns:
                series = weather_df[var]
            elif var in interaction_df.columns:
                series = interaction_df[var]
            else:
                continue

            advanced_lag_frames.append(
                series.shift(lag).rename(f"{var}_lag{lag}")
            )

    if advanced_lag_frames:
        advanced_lag_df = pd.concat(advanced_lag_frames, axis=1)
        print(f"   - Added {len(advanced_lag_df.columns)} advanced lag features")
    else:
        advanced_lag_df = pd.DataFrame(index=df_fe.index)
        print("   - No advanced lag features added")

    df_advanced = pd.concat([
        df_fe,
        volatility_df,
        weather_df,
        interaction_df,
        statistical_df,
        temporal_df,
        advanced_lag_df,
    ], axis=1)

    df_advanced = df_advanced.dropna().reset_index(drop=True)

    feature_categories = {
        "Original": len(df_fe.columns),
        "Volatility": len(volatility_df.columns),
        "Weather Patterns": len(weather_df.columns),
        "Interactions": len(interaction_df.columns),
        "Statistical": len(statistical_df.columns),
        "Temporal": len(temporal_df.columns),
        "Advanced Lags": len(advanced_lag_df.columns),
    }
    for category, count in feature_categories.items():
        print(f"   - {category}: {count} features")

    return df_advanced

def engineer_t2m_features_for_predict(
    input_path: str,
    output_path: str,
    lags: List[int] = None,
    windows: List[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:

    if lags is None:
        lags = LAG_LIST
    if windows is None:
        windows = ROLLING_WINDOWS

    df = load_and_sort_data(input_path)
    base_features = select_base_features(df, SELECTED_FEATURES)
    df_sel = build_base_frame(df, base_features)
    seasonal_df = add_seasonal_features(df_sel)
    lag_df = add_lag_features(df_sel, base_features, lags=lags)
    rolling_df = add_rolling_features(df_sel, rolling_cols=ROLLING_COLS, windows=windows)
    df_fe = pd.concat([
        df_sel,
        seasonal_df,
        lag_df,
        rolling_df,
    ], axis=1)

    df_fe = df_fe.dropna().reset_index(drop=True)
    df_final = add_advanced_features(df_fe)

    num_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != "date"]

    df_final.to_parquet(output_path, index=False)

    return df_final, feature_cols

def engineer_t2m_features_from_duckdb(
    duckdb_path: str,
    table_name: str,
    output_path: str,
    lags: List[int] = None,
    windows: List[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature Engineering pipeline โดยดึงข้อมูลจาก DuckDB table โดยตรง
    """
    
    con = duckdb.connect("md:Climate Change (T2M)")
    df = con.execute(f"SELECT * FROM {table_name}").df()
    con.close()

    if lags is None:
        lags = LAG_LIST
    if windows is None:
        windows = ROLLING_WINDOWS

    # 2) base features
    base_features = select_base_features(df, SELECTED_FEATURES)

    # 3) df_sel (ไม่มี target)
    df_sel = build_base_frame(df, base_features)

    # 4) seasonal
    seasonal_df = add_seasonal_features(df_sel)

    # 5) lag features
    lag_df = add_lag_features(df_sel, base_features, lags=lags)

    # 6) rolling features
    rolling_df = add_rolling_features(df_sel, rolling_cols=ROLLING_COLS, windows=windows)

    # 7) รวม base+seasonal+lag+rolling
    df_fe = pd.concat([
        df_sel,
        seasonal_df,
        lag_df,
        rolling_df,
    ], axis=1)

    # ลบ NaN จาก base lag/rolling ชุดแรก
    df_fe = df_fe.dropna().reset_index(drop=True)

    # 8) Advanced features (ใช้ df_fe ที่สะอาดแล้ว)
    df_final = add_advanced_features(df_fe)

    # 9) feature columns = ทุกคอลัมน์ตัวเลขที่ไม่ใช่ date
    num_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c != "date"]

    # เซฟออกเป็น Parquet
    df_final.to_parquet(output_path, index=False)

    return df_final, feature_cols