import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_feature_importance(importance_df, model_name, top_n=20):
    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Feature Importances - {model_name}',
        labels={'Importance': 'Importance', 'Feature': 'Feature'}
    )
    fig.update_layout(height=max(400, top_n * 25))
    return fig


def plot_predictions_vs_actual(y_true, y_pred, model_name):
    fig = px.scatter(
        x=y_true,
        y=y_pred,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title=f'Predicted vs Actual - {model_name}',
        opacity=0.6
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))

    return fig


def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'Residual Plot - {model_name}')

    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel('Residuals')
    ax2.set_title(f'Residual Distribution - {model_name}')

    plt.tight_layout()
    return fig


def plot_model_comparison(results_df):
    metrics = ['RMSE', 'MAE', 'R2']

    fig = go.Figure()

    for metric in metrics:
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric]
            ))

    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group'
    )

    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, model_name):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        name='Training Score',
        mode='lines+markers'
    ))

    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        name='Validation Score',
        mode='lines+markers'
    ))

    fig.update_layout(
        title=f'Learning Curve - {model_name}',
        xaxis_title='Training Set Size',
        yaxis_title='Score'
    )

    return fig


def plot_vif_results(vif_data):
    fig = px.bar(
        vif_data,
        x='VIF',
        y='Feature',
        orientation='h',
        title='Variance Inflation Factor (VIF) by Feature',
        labels={'VIF': 'VIF Score', 'Feature': 'Feature'}
    )

    fig.add_vline(x=10, line_dash="dash", line_color="red",
                  annotation_text="VIF Threshold = 10")

    fig.update_layout(height=max(400, len(vif_data) * 20))
    return fig1