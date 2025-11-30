import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .eda import get_correlation_with_target


def calculate_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df.corr()


def get_highly_correlated_features(df, threshold=0.5, target_col="SalePrice"):
    corr_matrix = calculate_correlation_matrix(df)

    highly_correlated = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                highly_correlated.append({
                    'Feature1': col1,
                    'Feature2': col2,
                    'Correlation': corr_value
                })

    return pd.DataFrame(highly_correlated)


def calculate_vif(df, target_col="SalePrice"):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if target_col in numeric_df.columns:
        features = numeric_df.drop(columns=[target_col])
    else:
        features = numeric_df

    vif_data = pd.DataFrame()
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(len(features.columns))]

    return vif_data.sort_values('VIF', ascending=False)


def select_features_by_vif(df, target_col="SalePrice", vif_threshold=10):
    features_to_keep = []
    df_temp = df.copy()

    while True:
        vif_data = calculate_vif(df_temp, target_col)
        max_vif = vif_data['VIF'].max()

        if max_vif > vif_threshold:
            feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            df_temp = df_temp.drop(columns=[feature_to_remove])
        else:
            break

    if target_col in df_temp.columns:
        features_to_keep = [col for col in df_temp.columns if col != target_col]
    else:
        features_to_keep = df_temp.columns.tolist()

    return features_to_keep, vif_data


def select_features_by_correlation(df, target_col="SalePrice", threshold=0.1, top_n=None):
    correlations = get_correlation_with_target(df, target_col)

    if correlations is not None:
        correlations = correlations[correlations.index != target_col]
        correlations_abs = correlations.abs()
        selected = correlations_abs[correlations_abs > threshold].sort_values(ascending=False)

        if top_n:
            selected = selected.head(top_n)

        return selected.index.tolist(), selected

    return [], None


def combine_feature_selection(df, target_col="SalePrice", corr_threshold=0.1, vif_threshold=10):
    corr_features, _ = select_features_by_correlation(df, target_col, corr_threshold)

    df_corr_selected = df[corr_features + [target_col]]
    vif_features, vif_data = select_features_by_vif(df_corr_selected, target_col, vif_threshold)

    return vif_features, vif_data