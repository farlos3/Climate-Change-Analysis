import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List


def prepare_climate_data(
    raw_parquet_path: str,
    output_parquet_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    quality_checks: bool = True
) -> dict:
    """
    Advanced Data Preparation: User's custom climate data preparation
    - Column normalization to lowercase  
    - Remove 39 unwanted columns
    - Handle missing flags (-999 variants)
    - Asian seasonal features
    - 7-day multi-horizon forecast targets (day 1-7)
    - NO StandardScaler (model doesn't require normalization)
    
    Args:
        raw_parquet_path: path ไฟล์ raw data
        output_parquet_path: path ไฟล์ output cleaned data
        start_date: วันเริ่มต้น (YYYY-MM-DD) หรือ None = ทั้งหมด
        end_date: วันสิ้นสุด (YYYY-MM-DD) หรือ None = ทั้งหมด  
        quality_checks: ทำ data quality checks หรือไม่
    
    Returns:
        dict: สถิติการ prepare data
    """
    
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Raw parquet not found: {raw_parquet_path}")
    
    print(f"🧹 ADVANCED DATA PREPARATION: Processing {raw_parquet_path}")
    
    # อ่านข้อมูล raw
    df = pd.read_parquet(raw_parquet_path)
    original_rows = len(df)
    original_cols = len(df.columns)
    print(f"   📊 Raw data: {original_rows:,} rows, {original_cols} columns")
    
    # 1. Column normalization to lowercase
    df.columns = df.columns.str.lower()
    print(f"   🔤 Normalized column names to lowercase")
    
    # 2. Remove columns with any NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        df = df.drop(columns=nan_cols)
        print(f"   🗑️  Removed {len(nan_cols)} columns with NaN values")
    
    # 3. Remove 39 specific unwanted columns
    nan_cols2 = [
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
    
    unwanted_found = [col for col in nan_cols2 if col in df.columns]
    if unwanted_found:
        df = df.drop(columns=unwanted_found)
        print(f"   🗑️  Removed {len(unwanted_found)} unwanted columns")
    
    # 4. Date processing and filtering
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        raise ValueError("Invalid dates found in data")
    
    # Remove last 3 days (latest dates)
    last3 = df['date'].nlargest(3).tolist()
    df = df[~df['date'].isin(last3)]
    print(f"   📅 Removed last 3 days: {[d.date() for d in last3]}")
    
    # Apply date range filtering
    if start_date:
        df = df[df['date'] >= start_date]
        print(f"   📅 Filtered from: {start_date}")
    
    if end_date:
        df = df[df['date'] <= end_date]
        print(f"   📅 Filtered to: {end_date}")
    
    # 5. Handle missing flags (-999 variants)
    missing_flags = [-999, -999.0, -9999, -99, -99.0]
    missing_stats = {}
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        mask = df[col].isin(missing_flags)
        if mask.any():
            count_missing = mask.sum()
            mean_val = df.loc[~mask, col].mean()
            df.loc[mask, col] = mean_val
            missing_stats[col] = count_missing
    
    if missing_stats:
        total_replaced = sum(missing_stats.values())
        print(f"   🔧 Replaced {total_replaced} missing flags with column means")
    
    # 6. Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # 7. Add time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.isocalendar().week
    df['weekday'] = df['date'].dt.dayofweek  # Monday = 0
    
    # 8. Asian seasonal features
    def asia_season(m):
        if m in [12, 1, 2]:
            return "winter"
        elif m in [3, 4, 5]:
            return "summer"
        else:
            return "rainy"
    
    df['season'] = df['month'].apply(asia_season)
    season_map = {"winter": 0, "summer": 1, "rainy": 2}
    df['season_num'] = df['season'].map(season_map)
    
    # Drop temporary season column
    df = df.drop(columns=["season"], errors='ignore')
    print(f"   🌏 Added Asian seasonal features (winter/summer/rainy)")
    
    # 9. Remove additional unwanted columns
    cols_to_drop = ['frost_days', 'precsno', 'precsnoland', 'snodp', 'frsno', 'frseaice']
    remaining_drops = [col for col in cols_to_drop if col in df.columns]
    if remaining_drops:
        df = df.drop(columns=remaining_drops)
        print(f"   🗑️  Removed {len(remaining_drops)} additional unwanted columns")
    
    # 10. Calculate ET total
    if 'evland' in df.columns and 'evptrns' in df.columns:
        df["et_total"] = df["evland"] + df["evptrns"]
        print(f"   ⚡ Calculated ET total from evland + evptrns")
    
    # 11. Create 7-day multi-horizon forecast targets (daily forecasts 1-7)
    H = 7  # forecast horizon (7 days)
    target_mapping = {
        "t2m": "t2m",
        "rain": "prectotcorr", 
        "et": "et_total",
        "soil": "gwettop",
        "wind": "ws10m"
    }
    
    forecast_targets = []
    for name, source_col in target_mapping.items():
        if source_col in df.columns:
            # Create daily forecasts for day 1 to day 7
            for h in range(1, H+1):
                target_col = f"{name}_d{h}_forecast"
                df[target_col] = df[source_col].shift(-h)
                forecast_targets.append(target_col)
    
    print(f"   🎯 Created {len(forecast_targets)} forecast targets (7-day multi-horizon)")
    
    # 12. Remove rows with missing forecast targets
    df = df.dropna(subset=forecast_targets)
    print(f"   🧹 Removed rows with missing forecast targets")
    
    # 13. Prepare features and targets (NO STANDARDSCALER)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in num_cols if col not in forecast_targets]
    
    print(f"   📊 Features: {len(feature_cols)} columns, Targets: {len(forecast_targets)} columns")
    print(f"   ⚠️  NO StandardScaler applied (model doesn't require normalization)")
    
    # 15. Save cleaned and prepared data
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path, index=False)
    
    final_rows = len(df)
    final_cols = len(df.columns)
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    
    # Summary statistics
    result = {
        'status': 'success',
        'operation': 'advanced_data_preparation',
        'input_file': raw_parquet_path,
        'output_file': output_parquet_path,
        'original_rows': original_rows,
        'original_cols': original_cols,
        'final_rows': final_rows,
        'final_cols': final_cols,
        'rows_removed': original_rows - final_rows,
        'cols_removed': original_cols - final_cols,
        'date_range': date_range,
        'forecast_targets': forecast_targets,
        'feature_columns': len(feature_cols),
        'missing_flags_replaced': sum(missing_stats.values()) if missing_stats else 0,
        'normalization': 'none (model doesn\'t require)'
    }
    
    print(f"✅ Advanced data preparation completed:")
    print(f"   📊 Final: {final_rows:,} rows, {final_cols} columns")
    print(f"   🗑️  Removed: {original_rows - final_rows:,} rows, {original_cols - final_cols} columns")
    print(f"   📅 Range: {date_range}")
    print(f"   🎯 Targets: {len(forecast_targets)} forecast columns (7-day multi-horizon)")
    print(f"   🚫 No normalization applied")
    print(f"   💾 Saved: {output_parquet_path}")
    
    return result


def prepare_climate_incremental(
    existing_clean_path: str,
    new_raw_path: str,
    output_clean_path: str,
    overlap_days: int = 3
) -> dict:
    """
    Incremental preparation with advanced data preparation logic
    - Applies user's custom preparation to new data
    - Maintains consistency with existing prepared data
    - NO StandardScaler (model doesn't require normalization)
    
    Args:
        existing_clean_path: path ไฟล์ clean data เก่า
        new_raw_path: path ไฟล์ raw data ใหม่
        output_clean_path: path ไฟล์ output
        overlap_days: จำนวนวันที่อนุญาตให้ overlap
    
    Returns:
        dict: สถิติการ incremental preparation
    """
    
    print(f"🔄 INCREMENTAL ADVANCED PREPARATION:")
    print(f"   📁 Existing: {existing_clean_path}")
    print(f"   📁 New raw: {new_raw_path}")
    
    # อ่านข้อมูลเก่า
    if os.path.exists(existing_clean_path):
        existing_df = pd.read_parquet(existing_clean_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'])
        latest_date = existing_df['date'].max()
        print(f"   📅 Existing data until: {latest_date.date()}")
    else:
        existing_df = pd.DataFrame()
        latest_date = None
        print(f"   📅 No existing data found")
    
    # Prepare new raw data with advanced preparation
    temp_prepared = new_raw_path.replace('.parquet', '_temp_prepared.parquet')
    
    # กำหนด start_date สำหรับ new data (overlap handling)
    if latest_date:
        start_date = (latest_date - timedelta(days=overlap_days)).strftime('%Y-%m-%d')
    else:
        start_date = None
    
    prepare_result = prepare_climate_data(
        raw_parquet_path=new_raw_path,
        output_parquet_path=temp_prepared,
        start_date=start_date,
        quality_checks=True
    )
    
    # อ่าน new prepared data
    new_df = pd.read_parquet(temp_prepared)
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    if not existing_df.empty:
        # รวมข้อมูล: ลบ overlap แล้วเพิ่มข้อมูลใหม่
        cutoff_date = new_df['date'].min()
        existing_filtered = existing_df[existing_df['date'] < cutoff_date]
        
        # รวมข้อมูล
        combined_df = pd.concat([existing_filtered, new_df], ignore_index=True)
        overlap_removed = len(existing_df) - len(existing_filtered)
        
        print(f"   🔄 Removed {overlap_removed} overlapping rows")
        print(f"   ➕ Added {len(new_df)} new rows")
    else:
        combined_df = new_df
        overlap_removed = 0
    
    # Sort และ save
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    combined_df.to_parquet(output_clean_path, index=False)
    
    # Cleanup temp file
    if os.path.exists(temp_prepared):
        os.remove(temp_prepared)
    
    result = {
        'status': 'success',
        'operation': 'incremental_advanced_preparation',
        'existing_rows': len(existing_df) if not existing_df.empty else 0,
        'new_rows_added': len(new_df),
        'overlap_rows_removed': overlap_removed,
        'final_rows': len(combined_df),
        'final_cols': len(combined_df.columns),
        'date_range': f"{combined_df['date'].min().date()} to {combined_df['date'].max().date()}",
        'output_file': output_clean_path,
        'preparation_stats': prepare_result
    }
    
    print(f"✅ Incremental advanced preparation completed:")
    print(f"   📊 Final: {len(combined_df):,} rows, {len(combined_df.columns)} columns")
    print(f"   📅 Range: {result['date_range']}")
    
    return result