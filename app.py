import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config

from src.data.loader import load_raw_data, get_missing_value_summary
from src.data.cleaner import clean_data_minimal, prepare_for_sklearn, get_cleaning_report
from src.analysis.eda import get_numerical_summary, get_categorical_summary, get_correlation_with_target
from src.models.trainer import split_data, train_all_models, get_feature_importance
from src.models.evaluator import evaluate_all_models
from src.models.predictor import predict_price, load_preprocessing_objects

st.set_page_config(page_title="House Price Analysis", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction Analysis")
st.markdown("Advanced regression analysis using multiple machine learning models")

if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None

tabs = st.tabs([
    "📊 EDA - Initial",
    "🧹 Data Preparation",
    "📈 Feature Analysis",
    "🤖 Models & Predictions"
])

with tabs[0]:
    st.header("Exploratory Data Analysis - Initial")

    df_raw = load_raw_data()

    if df_raw is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df_raw.shape[0])
        with col2:
            st.metric("Total Columns", df_raw.shape[1])
        with col3:
            st.metric("Missing Values", df_raw.isnull().sum().sum())

        st.subheader("Dataset Preview")
        st.dataframe(df_raw.head(10))

        st.subheader("Missing Values Analysis")
        missing_summary = get_missing_value_summary(df_raw)
        if len(missing_summary) > 0:
            st.dataframe(missing_summary)

            fig = px.bar(missing_summary,
                         x='Missing_Percentage',
                         y='Column',
                         orientation='h',
                         title='Missing Values by Feature',
                         labels={'Missing_Percentage': 'Missing %', 'Column': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Target Variable: SalePrice")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Statistics:**")
            st.write(df_raw['SalePrice'].describe())

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df_raw['SalePrice'], bins=50, kde=True, color='green', ax=ax)
            ax.set_title('Distribution of SalePrice')
            ax.set_xlabel('Sale Price')
            st.pyplot(fig)

        st.subheader("Numerical Features Distribution")
        numeric_cols = df_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'SalePrice'][:12]

        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            axes = axes.flatten()

            for idx, col in enumerate(numeric_cols):
                sns.histplot(df_raw[col].dropna(), bins=30, ax=axes[idx], kde=True)
                axes[idx].set_title(col)

            plt.tight_layout()
            st.pyplot(fig)

with tabs[1]:
    st.header("Data Preparation")

    df_raw = load_raw_data()

    if df_raw is not None:
        st.info("""
        This approach uses **minimal cleaning** (similar to TFDF notebook):
        - Removes only the 'Id' column
        - Keeps missing values (models can handle them)
        - No aggressive outlier removal
        - Encoding happens only when preparing for sklearn models
        """)

        if st.button("Prepare Data (Minimal Cleaning)", type="primary"):
            with st.spinner("Preparing data..."):
                df_cleaned = clean_data_minimal(df_raw)
                st.session_state['cleaned_df'] = df_cleaned

                report = get_cleaning_report(df_raw, df_cleaned)

                st.success("Data prepared successfully!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", report['rows_after'])
                    st.metric("Columns", report['columns_after'])
                with col2:
                    st.metric("Rows Removed", report['rows_removed'])
                with col3:
                    st.metric("Missing Values", report['missing_after'])

        if st.session_state['cleaned_df'] is not None:
            st.subheader("Prepared Data Preview")
            st.dataframe(st.session_state['cleaned_df'].head(10))

with tabs[2]:
    st.header("Feature Analysis")

    df = st.session_state.get('cleaned_df')
    if df is None:
        df = load_raw_data()
        if df is not None and 'Id' in df.columns:
            df = df.drop(columns=['Id'])

    if df is not None:
        st.subheader("Correlation with Target (SalePrice)")

        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if 'SalePrice' in numeric_df.columns:
            correlations = numeric_df.corr()['SalePrice'].sort_values(ascending=False)
            correlations = correlations[correlations.index != 'SalePrice']

            top_n = st.slider("Number of top features to show", 5, 30, 15)
            top_corr = correlations.head(top_n)

            fig = px.bar(x=top_corr.values,
                         y=top_corr.index,
                         orientation='h',
                         title=f'Top {top_n} Features Correlated with SalePrice',
                         labels={'x': 'Correlation', 'y': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlation Heatmap (Top Features)")
            top_features = top_corr.head(15).index.tolist() + ['SalePrice']
            corr_matrix = numeric_df[top_features].corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, ax=ax)
            ax.set_title('Correlation Heatmap - Top Features')
            st.pyplot(fig)

            st.session_state['top_features'] = top_corr.head(20).index.tolist()

with tabs[3]:
    st.header("Models & Predictions")

    df = st.session_state.get('cleaned_df')
    if df is None:
        df = load_raw_data()
        if df is not None and 'Id' in df.columns:
            df = df.drop(columns=['Id'])

    if df is not None:
        model_tabs = st.tabs(["Train Models", "Compare Results", "Make Predictions"])

        with model_tabs[0]:
            st.subheader("Train Regression Models")

            st.info("""
            **Models to train:**
            - **Linear Regression**: Simple baseline model
            - **Random Forest**: Ensemble of decision trees
            - **XGBoost**: Gradient boosting (optimized)
            """)

            use_top_features = st.checkbox("Use only top correlated features", value=True)

            if st.button("Train All Models", type="primary"):
                with st.spinner("Training models..."):
                    if use_top_features and 'top_features' in st.session_state:
                        features = st.session_state['top_features']
                        st.write(f"Using {len(features)} top features")
                    else:
                        features = [col for col in df.columns if col != 'SalePrice']

                    df_model, encoders = prepare_for_sklearn(df[features + ['SalePrice']])

                    X_train, X_test, y_train, y_test = split_data(df_model)

                    st.write(f"✓ Training set: {len(X_train)} samples")
                    st.write(f"✓ Test set: {len(X_test)} samples")

                    models = train_all_models(X_train, y_train)
                    results_df, predictions = evaluate_all_models(models, X_test, y_test)

                    st.session_state['models'] = models
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test
                    st.session_state['predictions'] = predictions
                    st.session_state['results_df'] = results_df
                    st.session_state['feature_names'] = features

                    st.success("✓ All models trained successfully!")
                    st.dataframe(results_df)

        with model_tabs[1]:
            st.subheader("Model Performance Comparison")

            if 'results_df' in st.session_state:
                results_df = st.session_state['results_df']

                st.dataframe(results_df, use_container_width=True)

                fig = go.Figure()
                for metric in ['RMSE', 'MAE', 'R2']:
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
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Detailed Model Analysis")
                model_choice = st.selectbox("Select Model", st.session_state['models'].keys(), key="compare_model")

                y_pred = st.session_state['predictions'][model_choice]
                y_test = st.session_state['y_test']

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.scatter(x=y_test, y=y_pred,
                                     labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                                     title=f'Predicted vs Actual - {model_choice}')

                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                             mode='lines', name='Perfect Prediction',
                                             line=dict(color='red', dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    residuals = y_test - y_pred
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(residuals, bins=50, kde=True, ax=ax)
                    ax.set_title(f'Residual Distribution - {model_choice}')
                    ax.set_xlabel('Residuals')
                    st.pyplot(fig)

                if model_choice in ['Random Forest', 'XGBoost']:
                    st.subheader("Feature Importance")
                    model = st.session_state['models'][model_choice]
                    importance_df = get_feature_importance(model, st.session_state['feature_names'])

                    if importance_df is not None:
                        top_n = min(20, len(importance_df))
                        top_imp = importance_df.head(top_n)

                        fig = px.bar(top_imp, x='Importance', y='Feature',
                                     orientation='h',
                                     title=f'Top {top_n} Feature Importances - {model_choice}')
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please train models first in the 'Train Models' tab")

        with model_tabs[2]:
            st.subheader("Make Price Predictions")

            if 'models' in st.session_state:
                st.write("Enter feature values to predict house price:")

                feature_names = st.session_state['feature_names']

                input_data = {}
                cols_per_row = 3

                for i in range(0, len(feature_names), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(feature_names):
                            feature = feature_names[i + j]
                            with col:
                                input_data[feature] = st.number_input(
                                    feature,
                                    value=0.0,
                                    key=f"input_{feature}"
                                )

                model_choice = st.selectbox("Select Model",
                                            ['Linear Regression', 'Random Forest', 'XGBoost'],
                                            key="predict_model")

                if st.button("Predict Price", type="primary"):
                    model = st.session_state['models'][model_choice]
                    scaler, encoders = load_preprocessing_objects()

                    prediction = predict_price(model, input_data, scaler, encoders)

                    st.success(f"### Predicted Price: ${prediction:,.2f}")

                    st.write("**All Model Predictions:**")
                    all_preds = {}
                    for name, mdl in st.session_state['models'].items():
                        pred = predict_price(mdl, input_data, scaler, encoders)
                        all_preds[name] = f"${pred:,.2f}"

                    st.dataframe(pd.DataFrame(list(all_preds.items()),
                                              columns=['Model', 'Predicted Price']))
            else:
                st.info("Please train models first")

st.sidebar.title("📋 Navigation")
st.sidebar.info("""
**App Features:**
- Exploratory Data Analysis
- Minimal data cleaning (TFDF approach)
- Feature correlation analysis
- Multiple ML models
- Interactive predictions
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
df_raw = load_raw_data()
if df_raw is not None:
    st.sidebar.metric("Rows", df_raw.shape[0])
    st.sidebar.metric("Features", df_raw.shape[1])
    st.sidebar.metric("Missing", df_raw.isnull().sum().sum())