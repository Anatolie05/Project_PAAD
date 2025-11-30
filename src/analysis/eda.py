import pandas as pd
import numpy as np


def get_numerical_summary(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return df[numeric_cols].describe()


def get_categorical_summary(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    summary = {}

    for col in categorical_cols:
        summary[col] = {
            'unique_values': df[col].nunique(),
            'top_value': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'top_frequency': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
        }

    return pd.DataFrame(summary).T


def get_distribution_stats(df, column):
    stats = {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'skewness': df[column].skew(),
        'kurtosis': df[column].kurtosis()
    }
    return stats


def get_correlation_with_target(df, target_col="SalePrice", top_n=20):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if target_col in numeric_df.columns:
        correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
        return correlations.head(top_n)
    return None


def detect_outliers_iqr(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return {
        'n_outliers': len(outliers),
        'percentage': (len(outliers) / len(df)) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def get_missing_patterns(df):
    missing_matrix = df.isnull()
    patterns = missing_matrix.sum(axis=1).value_counts().sort_index()
    return patterns


def get_value_counts(df, column, top_n=10):
    return df[column].value_counts().head(top_n)