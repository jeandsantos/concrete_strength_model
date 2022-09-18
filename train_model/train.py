# Load Packages
import os
import numpy as np
import warnings
import logging
import datetime
import mlflow, mlflow.sklearn

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

# import settings as s

from config.core import config, PATH_MODELS

from preprocessing.preprocessor import preprocessor
from utils.data_manager import DataManager, save_pipeline
from utils.scoring import scorers, calculate_scores
from learners import model_objects

np.random.RandomState(config.config_model.seed)

def main():

    ts = datetime.datetime.now()

    # Import and split data
    dm = DataManager(
        path=config.config_app.path_data, 
        columns_mapping=config.config_model.cols_mapping
        )

    X_train, X_test, y_train, y_test = dm\
        .load_dataset()\
        .split_data(
            col_features=config.config_model.cols_features,
            col_target=config.config_model.cols_target,
            test_size=config.config_model.test_size,
            seed=config.config_model.seed,
            col_stratify=config.config_model.stratify_by
        )
        
    # Placeholder for best-performing model and metric
    best_model = None
    best_model_metric = np.inf

    # TODO set logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)

    # Use mlflow to track experiments
    os.environ.setdefault(key='mlflow_tracking_uri', value=config.config_app.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.config_app.mlflow_tracking_uri)
    
    warnings.filterwarnings('ignore')
    ml_exp = mlflow.get_experiment_by_name(config.config_app.mlflow_experiment_name)
    
    if ml_exp is not None:
        # Delete previous experiment
        mlflow.delete_experiment(ml_exp.experiment_id)
        # empty_dir(Path.cwd() / 'mlruns' / '.trash')
        
    experiment_id = mlflow.create_experiment(
        f'{config.config_app.mlflow_experiment_name}_{ts.strftime(r"%Y%m%d_%H%m")}', 
        artifact_location='./artifacts',
        tags={'project': config.config_app.project_name}
    )
    
    print(f'Created Experiment \'{config.config_app.mlflow_experiment_name}\' | ID: {experiment_id}')

    mlflow.sklearn.autolog()

    # Train models
    for model_name in model_objects.keys():

        warnings.filterwarnings('ignore')
                
        if config.config_app.mlflow_bool:
            mlflow.start_run(
                experiment_id=experiment_id, 
                run_name=model_name+ts.strftime(r'_%Y%m%d_%H%m'), 
                tags={'model':model_name}
                )
            ml_run = mlflow.active_run()
        
        if config.config_app.bool_verbose:
            print('_'*50)
            print(f'{model_name}\n{ts}')

        # Pipeline for feature engineering and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_objects[model_name]['model'])
            ])

        # Hyperparameter tuning with k-fold cross-validation
        rs = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_objects[model_name]['params'],
            n_iter=config.config_model.search_iterations,
            random_state=config.config_model.seed,
            cv=config.config_model.cv_folds,
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
            n_jobs=config.config_model.n_jobs,
            )

        rs.fit(X_train, y_train)
            
        all_scores = {}

        for y_obs, x, suffix in [(y_train, X_train, '_train'), (y_test, X_test, '_test')]:
            
            scores = calculate_scores(y_obs, rs.predict(x), scorers, verbose=config.config_app.bool_verbose, suffix=suffix)
            all_scores.update(scores)
            
        save_pipeline(pipeline_to_persist=rs, file_path=PATH_MODELS / f'model_{model_name}.pkl')
        
        if all_scores['rmse_test'] < best_model_metric:
            
            best_model = model_name
            best_model_metric = all_scores['rmse_test']
            # Export model
            save_pipeline(pipeline_to_persist=rs, file_path=PATH_MODELS / 'best_model.pkl')
        
        if config.config_app.bool_verbose:
            print('Best Parameters for {} model:\n {}'.format(model_name, rs.best_params_))

            tts = ts.now() - ts
            print(f'Best model: {best_model}')
            print(f'Training time {round(tts.total_seconds())}s')
        
        mlflow.log_param('cv_folds', config.config_model.cv_folds)
        mlflow.log_param('cv_metric', config.config_model.cv_metric)
        mlflow.log_param('search_iterations', config.config_model.search_iterations)
        mlflow.log_param('seed', config.config_model.seed)
        mlflow.log_param('training_set_size', X_train.shape[0])
        mlflow.log_param('test_set_size', X_test.shape[0])    
        mlflow.log_param('best_params', rs.best_params_)
        
        mlflow.set_tag('model_type', model_name)
        mlflow.sklearn.log_model(rs, 'model')
        
        for param, value in rs.best_params_.items():
                mlflow.log_param(param.split('__')[-1], value)
        
        for metric, score in all_scores.items():
            mlflow.log_metric(metric, score)
        
        mlflow.end_run()

if __name__ == '__main__':
    main()