import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_missing_values(df):
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

    if len(missing_data) > 0:
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            labels={'x': 'Number of Missing Values', 'y': 'Features'},
            title='Missing Values by Feature'
        )
        fig.update_layout(height=max(400, len(missing_data) * 20))
        return fig
    return None


def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap')
    return fig


def plot_distribution(df, column):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[column], kde=True, ax=ax1)
    ax1.set_title(f'Distribution of {column}')
    ax1.set_xlabel(column)

    sns.boxplot(y=df[column], ax=ax2)
    ax2.set_title(f'Boxplot of {column}')

    plt.tight_layout()
    return fig


def plot_categorical_distribution(df, column, top_n=10):
    value_counts = df[column].value_counts().head(top_n)

    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={'x': column, 'y': 'Count'},
        title=f'Top {top_n} Categories in {column}'
    )
    return fig


def plot_correlation_heatmap(df, figsize=(14, 12)):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    return fig


def plot_correlation_with_target(correlations, target_col="SalePrice", top_n=20):
    corr_data = correlations.head(top_n)

    fig = px.bar(
        x=corr_data.values,
        y=corr_data.index,
        orientation='h',
        labels={'x': f'Correlation with {target_col}', 'y': 'Features'},
        title=f'Top {top_n} Features Correlated with {target_col}'
    )
    return fig


def plot_scatter_with_target(df, feature, target_col="SalePrice"):
    fig = px.scatter(
        df,
        x=feature,
        y=target_col,
        title=f'{feature} vs {target_col}',
        labels={feature: feature, target_col: target_col},
        opacity=0.6
    )
    return fig


def plot_numeric_distributions_grid(df, columns=None, ncols=3):
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns[:12]

    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    for idx, col in enumerate(columns):
        if idx < len(axes):
            sns.histplot(df[col], kde=True, ax=axes[idx])
            axes[idx].set_title(f'{col}')

    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_outliers_summary(outlier_stats):
    df_outliers = pd.DataFrame(outlier_stats).T
    df_outliers = df_outliers.sort_values('percentage', ascending=False)

    fig = px.bar(
        x=df_outliers.index,
        y=df_outliers['percentage'],
        labels={'x': 'Features', 'y': 'Outlier Percentage (%)'},
        title='Outlier Percentage by Feature'
    )
    return fig