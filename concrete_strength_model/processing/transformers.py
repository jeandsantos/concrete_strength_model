import typing as t
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class SumTransformer(BaseEstimator, TransformerMixin):
    """
    Create feature with the sum of the values of selected columns
    """
    
    def __init__(self, columns: t.Iterable, col_name: str) -> None:
        
        if not isinstance(columns, list):
            raise ValueError('columns must be a list')
        
        self.columns = columns
        self.col_name = col_name
        
    def fit(self, X:pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        
        values = X[self.columns].sum(axis=1)

        X[self.col_name] = values
        
        return X

     
class PercentageTransformer(BaseEstimator, TransformerMixin):
    """
    Converts counts to percentages
    """
    
    def __init__(self, col_numerator: t.List, col_denominator: t.List = None) -> None:
        
        if not isinstance(col_numerator, list):
            raise ValueError('col_numerator must be a list')
        
        self.col_numerator = col_numerator
        self.col_denominator = col_denominator
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        
        if (self.col_denominator is None):
            denomitator = X[self.col_numerator].sum(axis=1)
            
        elif isinstance(self.col_denominator, list):
            denomitator = X[self.col_denominator].sum(axis=1)

        X[self.col_numerator] = X[self.col_numerator].div(denomitator, axis=0).mul(100)

        return X

class RatioTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates ratios between one or more columns of a DataFrame
    """
    
    def __init__(self, col_numerator: t.List, col_denominator: t.List, name: str) -> None:
        
        if not isinstance(col_numerator, list):
            raise ValueError('col_numerator must be a list')
        
        if not isinstance(col_denominator, list):
            raise ValueError('col_denominator must be a list')
        
        self.col_numerator = col_numerator
        self.col_denominator = col_denominator
        self.name = name
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        
        X = X.copy()
        
        numerator = X[self.col_numerator].sum(axis=1)
        denomitator = X[self.col_denominator].sum(axis=1)

        X[self.name] = np.divide(numerator, denomitator)

        return X