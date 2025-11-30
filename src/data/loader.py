import pandas as pd
import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config


@st.cache_data
def load_raw_data():
    """
    Load raw data from CSV file

    Returns:
        pd.DataFrame: Raw dataset
    """
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {config.RAW_DATA_PATH}")
        st.info("Please ensure train.csv is in the data/raw/ directory")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data
def load_cleaned_data():
    """
    Load cleaned/processed data from CSV file

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    try:
        df = pd.read_csv(config.CLEANED_DATA_PATH)
        return df
    except FileNotFoundError:
        st.warning("Cleaned data not found. Please run data cleaning first.")
        return None
    except Exception as e:
        st.error(f"Error loading cleaned data: {str(e)}")
        return None


def get_data_info(df):
    """
    Get basic information about the dataset

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        dict: Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes,
        'missing_values': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2  # MB
    }
    return info


def get_basic_statistics(df):
    """
    Get basic statistical summary of the dataset

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Statistical summary
    """
    return df.describe()


def get_missing_value_summary(df):
    """
    Get detailed missing value analysis

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Missing value summary
    """
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values
    })

    # Sort by missing percentage
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    ).reset_index(drop=True)

    return missing_df