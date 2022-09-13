# A feature engineering pipeline for the transformation of both numeric and non-numeric data

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection

from preprocessing.transformers.numeric import SumTransformer, PercentageTransformer, RatioTransformer

from settings import (
    COLS_COMPOSITION, 
    COLS_SOLIDS,
    COLS_RATIO_AGGREGATES_SOLIDS, 
    COLS_RATIO_CEMENT_WATER,
    FE_DROP_CONSTANT_PARAMS,
    FE_SMART_CORRELATION_PARAMS,
)

preprocessor = Pipeline(steps=[
    # Create percentages
    ('percentage', PercentageTransformer(col_numerator=COLS_COMPOSITION)),
    
    ## Feature Creation
    # Create sum of solids
    ('total_solids', SumTransformer(columns=COLS_SOLIDS, col_name='total_solids')),
    
    # Create ratio-based features
    ('ratio_aggregates_solids', RatioTransformer(
        col_numerator=COLS_RATIO_AGGREGATES_SOLIDS[0], 
        col_denominator=COLS_RATIO_AGGREGATES_SOLIDS[1], 
        name='ratio_aggregates_solids')),
    # Create ratio (married to not married)
    ('ratio_cement_water', RatioTransformer(
        col_numerator=COLS_RATIO_CEMENT_WATER[0], 
        col_denominator=COLS_RATIO_CEMENT_WATER[1], 
        name='ratio_cement_water')),
    
    ## Feature Transformation
    # Yeo-Johnson Transform
    ('yeojohnson', YeoJohnsonTransformer()),
    # Z-score scaling
    ('standardization', SklearnTransformerWrapper(transformer=StandardScaler())),
    # Remove highly correlated 
    ('remove_correlated', SmartCorrelatedSelection(**FE_SMART_CORRELATION_PARAMS)),

    ## Feature Selection
    # Drop Constant Features
    ('drop_constant', DropConstantFeatures(**FE_DROP_CONSTANT_PARAMS)),
    ])
