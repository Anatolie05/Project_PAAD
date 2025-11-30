import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def clean_data_minimal(df):
    df_cleaned = df.copy()

    if 'Id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['Id'])

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(config.CLEANED_DATA_PATH, index=False)

    return df_cleaned


def prepare_for_sklearn(df, target_col='SalePrice'):
    df_prep = df.copy()

    numeric_cols = df_prep.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]
    categorical_cols = df_prep.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if df_prep[col].isnull().sum() > 0:
            df_prep[col].fillna(df_prep[col].median(), inplace=True)

    for col in categorical_cols:
        if df_prep[col].isnull().sum() > 0:
            df_prep[col].fillna('Missing', inplace=True)

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col] = le.fit_transform(df_prep[col].astype(str))
        encoders[col] = le

    config.PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ENCODERS_PATH, 'wb') as f:
        pickle.dump(encoders, f)

    df_prep = df_prep.replace([np.inf, -np.inf], np.nan)
    df_prep = df_prep.dropna()

    feature_cols = [col for col in df_prep.columns if col != target_col]
    with open(config.PREPROCESSING_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    return df_prep, encoders


def get_cleaning_report(df_before, df_after):
    report = {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'rows_removed': len(df_before) - len(df_after),
        'columns_before': len(df_before.columns),
        'columns_after': len(df_after.columns),
        'missing_before': df_before.isnull().sum().sum(),
        'missing_after': df_after.isnull().sum().sum(),
    }
    return report