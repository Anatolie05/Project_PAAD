import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return metrics, y_pred


def evaluate_all_models(models, X_test, y_test):
    results = []
    predictions = {}

    for model_name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test)
        metrics['Model'] = model_name
        results.append(metrics)
        predictions[model_name] = y_pred

    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'RMSE', 'MAE', 'R2']]

    return results_df, predictions


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_error_analysis(y_true, y_pred):
    residuals = y_true - y_pred

    analysis = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'mape': calculate_mape(y_true, y_pred)
    }

    return analysis