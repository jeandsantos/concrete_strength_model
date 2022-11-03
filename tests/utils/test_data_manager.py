import pytest
from concrete_strength_model.utils.data_manager import DataManager
from concrete_strength_model.config.core import PATH_MODELS, config


def test_data_manager():
    
    dm = DataManager(
        path=config.config_app.path_data,
        columns_mapping=config.config_model.cols_mapping,
    )
    
    assert isinstance(dm.path, str)
    assert isinstance(dm.columns_mapping, dict)