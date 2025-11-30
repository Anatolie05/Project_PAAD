"""
Configuration file for House Price Analysis Application
Contains paths, constants, and global settings
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "train.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_data.csv"

# Model paths
MODELS_DIR = BASE_DIR / "saved_models"
PREPROCESSING_DIR = MODELS_DIR / "preprocessing"

LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost.pkl"

SCALER_PATH = PREPROCESSING_DIR / "scaler.pkl"
ENCODERS_PATH = PREPROCESSING_DIR / "encoders.pkl"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PREPROCESSING_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature selection parameters
VIF_THRESHOLD = 10
CORRELATION_THRESHOLD = 0.5

# Target variable
TARGET_COLUMN = "SalePrice"

# Columns to potentially drop (Id column, etc.)
COLUMNS_TO_DROP = ["Id"]