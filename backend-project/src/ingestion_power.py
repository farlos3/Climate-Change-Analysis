# Incremental preparation logic moved from data_preparation.py
import numpy as np
from datetime import timedelta
import os
import requests
import pandas as pd
import datetime
from src.config import LAT, LON, PARAMETERS_AFTER_2001, DEFAULT_START, AFTER_2001_START, POWER_URL

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_power_daily_batch(param_file: str, output_parquet_path: str):

    params_df = pd.read_csv(param_file)
    valid_params = params_df["parameter"].tolist()

    # last day = today
    end_date = datetime.datetime.today().strftime("%Y%m%d")

    all_param_data = {}

    # Loop API calls in batches of 20 parameters
    for batch_idx, batch_params in enumerate(chunk_list(valid_params, 20), start=1):

        # If any parameter starts after 2001 → use new start date
        has_after_2001 = any(p in PARAMETERS_AFTER_2001 for p in batch_params)
        start_date = AFTER_2001_START if has_after_2001 else DEFAULT_START

        param_str = ",".join(batch_params)

        request_params = {
            "parameters": param_str,
            "community": "AG",
            "longitude": LON,
            "latitude": LAT,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }

        print(f"\n=== Batch {batch_idx} | {len(batch_params)} parameters ===")
        print(f"Parameters: {param_str}")
        print(f"Start date: {start_date}")

        try:
            r = requests.get(POWER_URL, params=request_params, timeout=60)
            print("Status:", r.status_code)

            if r.status_code == 422:
                print("422 Unprocessable Entity")
                print(r.text[:300])
                continue

            r.raise_for_status()
            data = r.json()
            records = data["properties"]["parameter"]

            for p_name, series_dict in records.items():
                all_param_data[p_name] = series_dict

            print(f"✓ Batch {batch_idx} collected {len(records)} params")

        except Exception as e:
            print(f"✗ Error in batch {batch_idx}: {e}")
            continue

    df_list = []
    for param, series_dict in all_param_data.items():
        s = pd.Series(series_dict, name=param)
        s.index = pd.to_datetime(s.index, format="%Y%m%d", errors="coerce")
        df_list.append(s)

    df = pd.concat(df_list, axis=1).sort_index()

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)

    # Convert index to column 'date' and save as Parquet
    df_out = df.reset_index().rename(columns={"index": "DATE"})
    df_out.to_parquet(output_parquet_path, index=False)

    print(f"\nSaved RAW Parquet to: {output_parquet_path}")
    print(f"Parquet shape: {df_out.shape}")

    return output_parquet_path