from pathlib import Path

URL_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

# Paths
PATH_PROJECT = Path.cwd()

# Lists of columns and features
COLS_MAPPING = {
    'Cement (component 1)(kg in a m^3 mixture)': 'cement',
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'slag',
    'Fly Ash (component 3)(kg in a m^3 mixture)': 'ash',
    'Water  (component 4)(kg in a m^3 mixture)': 'water',
    'Superplasticizer (component 5)(kg in a m^3 mixture)': 'superplasticizer',
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'coarse_aggregate',
    'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'fine_aggregate',
    'Age (day)': 'age',
    'Concrete compressive strength(MPa, megapascals) ': 'strength' 
}

COLS_FEATURES = [
    'cement', 'slag', 'ash',
    'water', 'superplasticizer', 'coarse_aggregate',
    'fine_aggregate', 'age',
    ]

COLS_COMPOSITION = [
    'cement', 'slag', 'ash', 
    'water', 'superplasticizer', 'coarse_aggregate', 
    'fine_aggregate',
    ]

COLS_SOLIDS = [col for col in COLS_COMPOSITION if col != 'water']

COLS_RATIO_AGGREGATES_SOLIDS = [
    ['coarse_aggregate', 'fine_aggregate',], 
    ['cement', 'slag', 'ash', 'superplasticizer',],
    ]

COLS_RATIO_CEMENT_WATER = [
    ['cement',], 
    ['water',],
    ]

COLS_TARGET = 'strength'

# Paramters for feature engineering
FE_DROP_CONSTANT_PARAMS={
    'tol':1,
    'missing_values':'ignore',
}

FE_SMART_CORRELATION_PARAMS={
    'threshold':0.90,
    'method':'pearson',
    'selection_method':'missing_values',
    'missing_values':'ignore',
}

# Other Parameters
BOOL_VERBOSE=True
SEED=1
N_JOBS=-2

# Parameters for model training, validation and tracking
TEST_SIZE=0.2
SEARCH_ITERATIONS=50

CV_SCORES=['mse', 'rmse', 'mae', 'mape', 'max_error', 'r2']
CV_METRIC='mse'
CV_FOLDS=10

MLFLOW_BOOL=True
MLFLOW_TRACKING_URI='http://localhost:5000'
MLFLOW_EXPERIMENT_NAME='concrete-mixtures'
