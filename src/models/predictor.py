import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
import config


def load_preprocessing_objects():
    try:
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    try:
        with open(config.ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
    except:
        encoders = None

    return scaler, encoders


def preprocess_input(input_data, scaler, encoders):
    try:
        with open(config.PREPROCESSING_DIR / 'feature_names.pkl', 'rb') as f:
            all_feature_names = pickle.load(f)
    except:
        all_feature_names = list(input_data.keys())

    input_dict = {}
    for feat in all_feature_names:
        if feat in input_data:
            input_dict[feat] = input_data[feat]
        else:
            input_dict[feat] = 0

    df_input = pd.DataFrame([input_dict])

    if encoders:
        for col, encoder in encoders.items():
            if col in df_input.columns:
                try:
                    df_input[col] = encoder.transform(df_input[col].astype(str))
                except:
                    df_input[col] = 0

    if scaler:
        df_input = pd.DataFrame(
            scaler.transform(df_input),
            columns=df_input.columns
        )

    return df_input


def predict_price(model, input_data, scaler=None, encoders=None):
    if scaler is None or encoders is None:
        scaler, encoders = load_preprocessing_objects()

    processed_input = preprocess_input(input_data, scaler, encoders)
    prediction = model.predict(processed_input)

    return prediction[0]