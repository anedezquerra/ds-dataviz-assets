# -*- coding:utf-8 -*-

from typing import List
import plotly.graph_objects as go
import pandas as pd
from catboost import CatBoostRegressor


def plot_residuals_vs_predicted(
        model: CatBoostRegressor,
        feature_names: List[str],
        data: pd.DataFrame,
        target_name: str,
        id_col: str = "id"
) -> go.Figure:
    """
    Create a residual plot (residuals vs predicted values) using Plotly for Dash.

    Parameters:
    - model: Trained CatBoostRegressor
    - feature_names: List of feature column names used for training
    - data: DataFrame including features, target, and ID column
    - target_name: Name of the target variable
    - id_col: Column name to use as observation ID for hover annotation

    Returns:
    - plotly.graph_objs._figure.Figure object
    """
    X = data[feature_names]
    y_true = data[target_name]
    y_pred = model.predict(X)
    residuals = y_true - y_pred
    ids = data[id_col].astype(str).values

    # Prepare hover text
    hover_text = [
        f"ID: {id_}<br>Predicted: {pred:.2f}<br>Residual: {res:.2f}<br>Actual: {act:.2f}"
        for id_, pred, res, act in zip(ids, y_pred, residuals, y_true)
    ]

    fig = go.Figure()

    # Scatter plot of residuals
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='rgba(255, 99, 71, 0.6)', size=6),
        text=hover_text,
        hoverinfo='text',
        name='Residuals'
    ))

    # Horizontal reference line at y = 0
    fig.add_trace(go.Scatter(
        x=[min(y_pred), max(y_pred)],
        y=[0, 0],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Zero Residual Line'
    ))

    fig.update_layout(
        title='Residuals vs Predicted Values',
        xaxis_title='Predicted Values',
        yaxis_title='Residuals (Actual - Predicted)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
    )

    return fig


if __name__ == '__main__':
    pass