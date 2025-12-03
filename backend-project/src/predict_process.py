import duckdb
import pandas as pd
import sys

def run_feature_selection(df):

    from src.predict import load_selected_features, prepare_features
    selected_features, _ = load_selected_features()
    df_selected = prepare_features(df, selected_features)
    return df_selected

def run_prediction(features_json):

    from src.predict import predict_all_models
    prediction_result = predict_all_models(features_json)
    return prediction_result

def update_actual_in_forecast_table(duckdb_path, table_name="forecast_store"):
    """
    Update actual values in forecast table using climate_clean (where actual data is available).
    Match by target_date and update actual for all models.
    """
    con = duckdb.connect(duckdb_path)
    # Get all target_dates in forecast table
    target_dates = con.execute(f"SELECT DISTINCT target_date FROM {table_name}").fetchall()
    target_dates = [row[0] for row in target_dates if row[0] is not None]
    if target_dates:
        date_str = ','.join([f"'{d}'" for d in target_dates])
        # Get actuals from climate_clean
        actuals = con.execute(f"SELECT date as target_date, T2M as actual FROM climate_clean WHERE date IN ({date_str})").fetchall()
        actual_dict = {row[0]: row[1] for row in actuals}
        # Update actual in forecast table
        for t_date, actual_val in actual_dict.items():
            con.execute(f"UPDATE {table_name} SET actual = {actual_val} WHERE target_date = '{t_date}'")
    con.close()
    print(f"Updated actual values in {table_name} for available target_dates.")

def save_forecast_to_duckdb(prediction_result, duckdb_path, table_name="forecast_store"):

    forecast_rows = []

    con = duckdb.connect(duckdb_path)
    last_date_row = con.execute("SELECT MAX(date) as last_date FROM climate_clean").fetchone()
    last_date = last_date_row[0] if last_date_row else None

    # horizon_day fix is 1-7
    forecast_rows = []
    import datetime
    base_date = pd.to_datetime(last_date)
    for model_name, rows in prediction_result.items():
        for idx, row in enumerate(rows):
            horizon_day = (idx % 7) + 1
            target_date = (base_date + datetime.timedelta(days=horizon_day)).strftime('%Y-%m-%d')

            forecast_value = None
            for key in ['forecast', 'T2M', 'prediction', 'value', 'y_pred', 'output']:
                if key in row:
                    forecast_value = row[key]
                    break

            if forecast_value is None:
                for v in row.values():
                    if isinstance(v, (int, float)):
                        forecast_value = v
                        break
            # debug log example row if forecast_value is still None
            if forecast_value is None:
                print(f"[DEBUG] Forecast value not found in row: {row}")
            forecast_rows.append({
                'date': last_date,
                'target_date': target_date,
                'horizon_day': horizon_day,
                'forecast': forecast_value,
                'actual': None,
                'model': model_name
            })
    df_forecast = pd.DataFrame(forecast_rows)

    # Fill actual from climate_clean:T2M by matching target_date
    if 'target_date' in df_forecast.columns:
        target_dates = df_forecast['target_date'].dropna().unique().tolist()
        if target_dates:
            date_str = ','.join([f"'{d}'" for d in target_dates])
            actual_df = con.execute(f"SELECT date as target_date, T2M as actual FROM climate_clean WHERE date IN ({date_str})").df()
            df_forecast = pd.merge(df_forecast, actual_df, on='target_date', how='left', suffixes=('', '_actual'))
            df_forecast['actual'] = df_forecast['actual_actual'].combine_first(df_forecast['actual'])
            df_forecast = df_forecast.drop(columns=['actual_actual'])

    # Register DataFrame as temp table
    con.register('df_forecast', df_forecast)
    # Create table if not exists (according to new schema)
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (date DATE, target_date DATE, horizon_day INTEGER, forecast DOUBLE, actual DOUBLE, model VARCHAR)")
    # Upsert: delete only rows with duplicate date, target_date, model then insert new
    for _, row in df_forecast.iterrows():
        con.execute(f"DELETE FROM {table_name} WHERE date = '{row['date']}' AND target_date = '{row['target_date']}' AND model = '{row['model']}'")
    con.execute(f"INSERT INTO {table_name} SELECT date, target_date, horizon_day, forecast, actual, model FROM df_forecast")
    con.close()
    print(f"Saved/updated forecast to DuckDB table: {table_name} ({len(df_forecast)} rows)")
    return df_forecast

