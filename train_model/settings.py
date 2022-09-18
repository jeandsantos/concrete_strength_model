from pathlib import Path

PATH_DATA = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

# Paths
PATH_PROJECT = Path.cwd()
PATH_MODELS = PATH_PROJECT / 'models'

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

cols_features = [
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

cols_target = 'strength'

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
seed=1

n_jobs=-2
N_FEATURES=len(cols_features)

# Parameters for model training, validation and tracking
test_size=0.2
STRATIFY_BY='age'
search_iterations=50

CV_SCORES=['mse', 'rmse', 'mae', 'mape', 'max_error', 'r2']
CV_METRIC='mse'
cv_folds=10

mlflow_bool=True
mlflow_tracking_uri='http://localhost:5000'
mlflow_experiment_name='concrete-mixtures'

PROJECT_NAME='concrete-mixture-optimization'