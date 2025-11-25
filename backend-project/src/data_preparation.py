import pandas as pd
import numpy as np
from typing import List, Tuple

MISSING_FLAGS = [-999, -999.0, -9999, -99, -99.0]

NAN_COLS_EXTRA = [
    'airmass',
    'allsky_kt',
    'allsky_nkt',
    'allsky_sfc_lw_dwn',
    'allsky_sfc_lw_up',
    'allsky_sfc_par_diff',
    'allsky_sfc_par_dirh',
    'allsky_sfc_par_tot',
    'allsky_sfc_sw_diff',
    'allsky_sfc_sw_dirh',
    'allsky_sfc_sw_dni',
    'allsky_sfc_sw_dwn',
    'allsky_sfc_sw_up',
    'allsky_sfc_uva',
    'allsky_sfc_uvb',
    'allsky_sfc_uv_index',
    'allsky_srf_alb',
    'aod_55',
    'aod_55_adj',
    'aod_84',
    'gwm_height',
    'gwm_height_anomaly',
    'imerg_precliquid_prob',
    'imerg_prectot',
    'imerg_prectot_count',
    'midday_insol',
    'original_allsky_sfc_lw_dwn',
    'original_allsky_sfc_sw_diff',
    'original_allsky_sfc_sw_dirh',
    'original_allsky_sfc_sw_dwn',
    'original_clrsky_sfc_lw_dwn',
    'original_clrsky_sfc_sw_dwn',
    'psh',
    'pw',
    'srf_alb_adj',
    'sza',
    'toa_sw_dni',
    'toa_sw_dwn',
    'ts_adj'
]

ZERO_COLS_TO_DROP = [
    'frost_days',
    'precsno',
    'precsnoland',
    'snodp',
    'frsno',
    'frseaice'
]

def check_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    นับจำนวนค่า missing flags (-999, -99, etc.) ในแต่ละคอลัมน์
    """
    rows = []
    for col in df.columns:
        count = df[col].isin(MISSING_FLAGS).sum()
        rows.append({"column_name": col, "missing_flag_count": count})
    return pd.DataFrame(rows)


def asia_season(month: int) -> str:
    """
    map เดือน → season แบบที่ใช้ในโน้ตบุ๊ก
    """
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    else:
        return "rainy"

def load_and_normalize_columns(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    df.columns = df.columns.str.lower()
    return df

def drop_nan_columns(df: pd.DataFrame) -> pd.DataFrame:

    nan_cols = df.columns[df.isna().any()].tolist()
    df = df.drop(columns=nan_cols)

    # ลบชุดที่ user ระบุเพิ่ม
    df = df.drop(columns=NAN_COLS_EXTRA, errors="ignore")
    return df

def handle_missing_flags_and_dates(df: pd.DataFrame) -> pd.DataFrame:

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values("date")

    # หา 3 วันที่ล่าสุด (เหมือนใช้ nlargest(3))
    last3 = df['date'].nlargest(3).tolist()
    
    # ลบ 3 วันที่ล่าสุดออก
    df = df[~df["date"].isin(last3)]

    # แทนค่าที่เป็น missing flag ด้วย mean ของคอลัมน์
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        mask = df[col].isin(MISSING_FLAGS)
        if mask.any():
            mean_val = df.loc[~mask, col].mean()
            df.loc[mask, col] = mean_val

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["week"] = df["date"].dt.isocalendar().week
    df["weekday"] = df["date"].dt.dayofweek # Monday = 0

    df["season"] = df["month"].apply(asia_season)
    season_map = {
        "winter": 0,
        "summer": 1,
        "rainy": 2
    }
    df["season_num"] = df["season"].map(season_map)

    # ลบ season (ตามโน้ตบุ๊ก)
    df = df.drop(columns=['season'], errors='ignore')

    return df


def drop_zero_dominated_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=ZERO_COLS_TO_DROP, errors="ignore")
    return df

def add_et_total(df: pd.DataFrame) -> pd.DataFrame:

    df['et_total'] = df['evland'] + df['evptrns']
    return df

def add_forecast_targets(df: pd.DataFrame, horizon: int = 7) -> Tuple[pd.DataFrame, List[str]]:

    target_cols = []

    # Temperature
    for h in range(1, horizon + 1):
        col_name = f"t2m_d{h}_forecast"
        df[col_name] = df["t2m"].shift(-h)
        target_cols.append(col_name)

    # Rain
    for h in range(1, horizon + 1):
        col_name = f"rain_d{h}_forecast"
        df[col_name] = df["prectotcorr"].shift(-h)
        target_cols.append(col_name)

    # ET
    for h in range(1, horizon + 1):
        col_name = f"et_d{h}_forecast"
        df[col_name] = df["et_total"].shift(-h)
        target_cols.append(col_name)

    # Soil moisture
    for h in range(1, horizon + 1):
        col_name = f"soil_d{h}_forecast"
        df[col_name] = df["gwettop"].shift(-h)
        target_cols.append(col_name)

    # Wind
    for h in range(1, horizon + 1):
        col_name = f"wind_d{h}_forecast"
        df[col_name] = df["ws10m"].shift(-h)
        target_cols.append(col_name)

    return df, target_cols


def select_feature_and_target_columns(df: pd.DataFrame, target_cols: List[str]) -> Tuple[List[str], List[str]]:

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in num_cols if col not in target_cols]
    return feature_cols, target_cols


# -----------------------------
# main pipeline
# -----------------------------
def prepare_nasa_power_data(
    input_path: str,
    output_path: str,
    # horizon: int = 7
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    
    df = load_and_normalize_columns(input_path)
    df = drop_nan_columns(df)
    df = handle_missing_flags_and_dates(df)
    df = add_time_features(df)
    df = drop_zero_dominated_columns(df)
    df = add_et_total(df)
    # df, target_cols = add_forecast_targets(df, horizon=horizon)
    # feature_cols, target_cols = select_feature_and_target_columns(df, target_cols)

    df.to_csv(output_path, index=False)
    return df

# if __name__ == "__main__":
    
#     INPUT_PATH = "backend-project/airflow/data/raw/power_daily1.parquet"
#     OUTPUT_PATH = "power_daily_prepared.csv"

#     df_prepared = prepare_nasa_power_data(
#         INPUT_PATH,
#         OUTPUT_PATH,
#         # horizon=7
#     )