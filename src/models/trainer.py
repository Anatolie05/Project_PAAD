import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def split_data(df, target_col=config.TARGET_COLUMN, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=config.RANDOM_STATE):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_STATE):
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None


def train_all_models(X_train, y_train):
    models = {}

    models['Linear Regression'] = train_linear_regression(X_train, y_train)
    save_model(models['Linear Regression'], config.LINEAR_MODEL_PATH)

    models['Random Forest'] = train_random_forest(X_train, y_train)
    save_model(models['Random Forest'], config.RF_MODEL_PATH)

    models['XGBoost'] = train_xgboost(X_train, y_train)
    save_model(models['XGBoost'], config.XGBOOST_MODEL_PATH)

    return models