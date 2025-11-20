TARGET_COL = 'tsurf'
SPLIT_DATE = '2022-01-01' 

COLS_TO_DROP_PRE_TRAIN = [    
    'season'     
]

XGB_PARAMS = {
    'objective': 'reg:squarederror', 
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42,
    'n_jobs': -1
}