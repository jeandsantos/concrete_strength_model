# Load Packages
import typing as t
import numpy as np
import pandas as pd
import warnings, logging, datetime, shutil
import mlflow, mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    make_scorer, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    mean_squared_error, 
    r2_score, 
    max_error
)
from sklearn.pipeline import Pipeline

import settings as s

from preprocessing.preprocessor import preprocessor
from utils.data_manager import DataManager, save_pipeline, empty_dir
from utils.scoring import scorers, calculate_scores
from learners.models import model_objects

np.random.RandomState(s.SEED)

def main():

    print('PATH: ', Path.cwd())
    print('PATH_PARENT: ', Path.cwd().parent)

    # Import and split data
    dm = DataManager(
        path=s.URL_DATA, 
        columns_mapping=s.COLS_MAPPING
        )

    X_train, X_test, y_train, y_test = dm\
        .load_dataset()\
        .split_data(
            col_features=s.COLS_FEATURES,
            col_target=s.COLS_TARGET,
            test_size=s.TEST_SIZE,
            seed=s.SEED,
            col_stratify=s.STRATIFY_BY
        )
        
    # Placeholder for best-performing model and metric
    best_model = None
    best_model_metric = np.inf

    # TODO set logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    # Use mlflow to track experiments
    if s.MLFLOW_BOOL:
            
        # TODO connect to sqlite db
        # os.environ.setdefault(key='MLFLOW_TRACKING_URI', value=MLFLOW_TRACKING_URI)
        # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        warnings.filterwarnings('ignore')
        ml_exp = mlflow.get_experiment_by_name(s.MLFLOW_EXPERIMENT_NAME)
        
        
        if ml_exp is not None:
            # Delete previous experiment
            mlflow.delete_experiment(ml_exp.experiment_id)
            empty_dir(Path.cwd() / 'mlruns' / '.trash')
            
        experiment_id = mlflow.create_experiment(
            s.MLFLOW_EXPERIMENT_NAME,
            tags={"project": s.PROJECT_NAME}
        )
        
        print(f'Created Experiment \'{s.MLFLOW_EXPERIMENT_NAME}\' | ID: {experiment_id}')

        mlflow.sklearn.autolog()

    # Train models
    for model_name in model_objects.keys():

        warnings.filterwarnings('ignore')
        ts = datetime.datetime.now()
        
        if s.MLFLOW_BOOL:
            mlflow.start_run(
                experiment_id=experiment_id, 
                run_name=model_name+ts.strftime(r'_%Y%m%d_%H%m'), 
                tags={'model':model_name}
                )
            ml_run = mlflow.active_run()
        
        if s.BOOL_VERBOSE:
            print('_'*50)
            print(model_name)
            print(ts)

        # Pipeline for feature engineering and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_objects[model_name]['model'])
            ])

        # Hyperparameter tuning with k-fold cross-validation
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_objects[model_name]['params'],
            n_iter=s.SEARCH_ITERATIONS,
            random_state=s.SEED,
            cv=s.CV_FOLDS,
            scoring={
                'mse': make_scorer(score_func=mean_squared_error, squared=True, greater_is_better=False),
                'rmse': make_scorer(score_func=mean_squared_error, squared=False, greater_is_better=False),
                'mae': make_scorer(score_func=mean_absolute_error, greater_is_better=False),
                'mape': make_scorer(score_func=mean_absolute_percentage_error, greater_is_better=False),
                'max_error': make_scorer(score_func=max_error, greater_is_better=False),
                'r2': make_scorer(score_func=r2_score, greater_is_better=True),
            },
            refit='mse',
            return_train_score=False,
            n_jobs=s.N_JOBS,
            )

        rs.fit(X_train, y_train)
            
        all_scores = {}

        for y_obs, x, suffix in [(y_train, X_train, '_train'), (y_test, X_test, '_test')]:
            
            scores = calculate_scores(y_obs, rs.predict(x), scorers, verbose=s.BOOL_VERBOSE, suffix=suffix)
            all_scores.update(scores)
            
        save_pipeline(pipeline_to_persist=rs, file_path=s.PATH_MODELS / f'model_{model_name}.pkl')
        
        if all_scores['rmse_test'] < best_model_metric:
            
            best_model = model_name
            best_model_metric = all_scores['rmse_test']
            # Export model
            save_pipeline(pipeline_to_persist = rs, file_path = s.PATH_MODELS / 'best_model.pkl')
        
        if s.BOOL_VERBOSE:
            print("Best Parameters for {} model:\n {}".format(model_name, rs.best_params_))

            tts = ts.now() - ts
            print(f'Best model: {best_model}')
            print(f'Training time {round(tts.total_seconds())}s')
        
        if s.MLFLOW_BOOL:
            
            mlflow.log_param("cv_folds", s.CV_FOLDS)
            mlflow.log_param("cv_metric", s.CV_METRIC)
            mlflow.log_param("search_iterations", s.SEARCH_ITERATIONS)
            mlflow.log_param("seed", s.SEED)
            mlflow.log_param("training_set_size", X_train.shape[0])
            mlflow.log_param("test_set_size", X_test.shape[0])    
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("best_params", rs.best_params_)
            
            for param, value in rs.best_params_.items():
                    mlflow.log_param(param.split('__')[-1], value)
            
            for metric, score in all_scores.items():
                mlflow.log_metric(metric, score)
            
            mlflow.sklearn.log_model(rs, 'model')
            mlflow.end_run()

if __name__ == '__main__':
    main()