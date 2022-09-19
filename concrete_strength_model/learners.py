# Contains model object and the parameters to be used for hyperparameter tuning
import numpy as np

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from config.core import config

model_objects = {
    'pls': {
        'model': PLSRegression(
            max_iter=1000
            ),
        'params': {
            'model__n_components': range(1, config.config_model.n_features),
        }
    },
    # 'lasso': {
    #     'model': Lasso(
    #         random_state=config.config_model.seed
    #     ),
    #     'params': {
    #         'model__max_iter': [1000, 5000,],
    #         'model__alpha': np.logspace(-6, 6, 13),
    #     }
    # },
    'elasticnet': {
        'model': ElasticNet(
            random_state=config.config_model.seed,
            max_iter=1000,
        ),
        'params': {
            'model__alpha': np.logspace(-6, 6, 13),
            'model__l1_ratio': [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0,],
        }
    },
    'svr': {
        'model': SVR(
            verbose=False
        ),
        'params': {
            'model__kernel': ['poly', 'rbf',],
            'model__degree': [2, 3, 4,],
            'model__C': np.logspace(-3, 3, 7),
            'model__tol': np.logspace(-3, 3, 7),
            'model__epsilon': np.logspace(-3, 3, 7),
        }
    },
    # 'random_forest': {
    #     'model': RandomForestRegressor(
    #         criterion='squared_error',
    #         n_jobs=config.config_model.n_jobs,
    #         random_state=config.config_model.seed,
    #         verbose=False,
    #         ),
    #     'params': {
    #         'model__n_estimators': [100, 250, 500,],
    #         'model__max_depth': [3, 5, 10, None,],
    #         'model__max_features': [0.3, 0.5, 0.7, None,],
    #         'model__min_samples_leaf': [5, 10, 25, 50,],
    #         }
    # },
    # 'gbm': {
    #     'model': GradientBoostingRegressor(
    #         loss='squared_error',
    #         random_state=config.config_model.seed,
    #         verbose=False,
    #         ),
    #     'params': {
    #         'model__n_estimators': [50, 100, 250,],
    #         'model__learning_rate': np.logspace(-3, 3, num=5),
    #         'model__max_depth': [3, 5, 10,],
    #         'model__max_features': [0.3, 0.7, None,],
    #         'model__min_samples_leaf': [1, 10,],
    #         'model__min_samples_split': [2,],
    #         }
    # },
    # 'mlp': {
    #     'model': MLPRegressor(
    #         solver='adam',
    #         n_iter_no_change=10,
    #         random_state=config.config_model.seed,
    #         verbose=False,
    #         ),
    #     'params': {
    #         'model__activation': ['relu', 'logistic', 'tanh'],
    #         'model__max_iter': [500, 1000, 2000],
    #         'model__hidden_layer_sizes': [(x,) for x in [5, 10, 20, 30,]] + [(x,x) for x in [5, 10, 20, 30,]],
    #         'model__learning_rate_init': np.logspace(-3, 3, 7),
    #         }
    # },
}