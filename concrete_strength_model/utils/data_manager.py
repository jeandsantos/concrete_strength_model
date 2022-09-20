import os
import shutil
import typing as t

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class DataManager:
    """
    A Class to import and transform data
    """

    def __init__(self, path: str, columns_mapping: dict) -> None:

        self.path = path
        self.columns_mapping = columns_mapping

    def load_dataset(self, *args, **kwargs):
        """Load Data and save it as attribute

        Returns:
            pd.DataFrame: a pandas dataframe of the imported data
        """

        self._df = pd.read_excel(self.path, *args, **kwargs)
        self._df = self._df.rename(columns=self.columns_mapping)

        return self

    def get_data(self) -> pd.DataFrame:

        return self._df

    def split_data(
        self,
        col_features: list,
        col_target: str = None,
        test_size: float = 0.2,
        seed: int = 1,
        col_stratify: str = None,
        *args,
        **kwargs
    ):

        if col_target is None:
            col_target = [col for col in self._df.columns if col not in col_features]

        if col_stratify is not None:
            col_stratify = self._df[col_stratify]

        # Split Data
        self._X = self._df[col_features]
        self._y = self._df[col_target]

        # Create training and test set
        X_train, X_test, y_train, y_test = train_test_split(
            self._X,
            self._y,
            test_size=test_size,
            random_state=seed,
            stratify=col_stratify,
            *args,
            **kwargs
        )

        return X_train, X_test, y_train, y_test

    @property
    def df(self):
        return self._df


def save_pipeline(pipeline_to_persist: Pipeline, file_path: str) -> None:
    """Saves the pipeline to a pkl file

    Args:
        pipeline_to_persist (Pipeline): pipeline to save
        file_path (str): file path to save the file
    """

    joblib.dump(pipeline_to_persist, file_path)


def load_pipeline(file_path: str) -> Pipeline:
    """Load a pipeline saved as a pkl file

    Args:
        file_path (str): file path of file

    Returns:
        Pipeline: a scikit-learn Pipeline object
    """

    trained_model = joblib.load(filename=file_path)

    return trained_model


def remove_old_pipelines(files_to_keep: t.List[str], path_models: str) -> None:
    """
    Removes old model pipelines
    """
    do_not_delete = files_to_keep + ["__init__.py", "training_data.csv"]
    for model_file in os.listdir(path_models):

        path_file = os.path.join(path_models, model_file)

        if (model_file not in do_not_delete) and (".png" not in path_file):
            os.remove(path_file)


def empty_dir(directory):
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
