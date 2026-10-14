
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler,OrdinalEncoder
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import numpy as np

def get_numeric_columns_indexes(df: pd.DataFrame):
    column_names = df.select_dtypes(include=['int64','float64']).columns
    return get_columns_indexes(df, column_names)

def get_category_columns_indexes(df: pd.DataFrame):
    column_names = df.select_dtypes(include=['object','category']).columns
    return get_columns_indexes(df, column_names)

def get_columns_indexes(df: pd.DataFrame, column_names):
    column_idxes = [df.columns.get_loc(col) for col in column_names]
    return column_idxes

def get_pipeline(df, model):
    num = get_numeric_columns_indexes(df)
    cat = get_category_columns_indexes(df)
    ordinal = []
    preprocessing = get_preprocessing(num, cat, ordinal)

    pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing), 
    ('regressor', model), 
    # ('smote', adasyn),
    ])
    return pipeline


def get_preprocessing(numeric_columns, categorical_columns, ordinal_columns):
    transformers = []
    if len(numeric_columns) > 0:
        handle_nummerical = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler()),
        ])
        transformers.append(('num', handle_nummerical, numeric_columns))
    if len(categorical_columns) > 0:
        handle_nomial = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ])
        transformers.append(('cat', handle_nomial, categorical_columns))
        
    if len(ordinal_columns) > 0:
        handle_ordinal = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OrdinalEncoder()),
        ])
        transformers.append(('ordinal', handle_ordinal, ordinal_columns))

    preprocessing = ColumnTransformer(transformers=transformers)
    return preprocessing


def prepare_features(df, FEATURES):
    # validate features
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Expected a nonempty DataFrame")

    # validate columns names
    if not df.columns.is_unique or not set(FEATURES).issubset(df.columns):
        raise ValueError("Expected exactly the four Iris features")

    ordered = df.loc[:, FEATURES]

    # validate column dtypes
    for dtype in ordered.dtypes:
        if not is_numeric_dtype(dtype) or is_bool_dtype(dtype):
            raise ValueError("Features must be numeric measurements")

    values = ordered.to_numpy(dtype=np.float64, copy=True)

    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")

    if (values <= 0).any():
        raise ValueError("Measurements must be positive")

    return values