# -*- coding:utf-8 -*-

from typing import List
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def plot_actual_vs_predicted_scatter(
        model: CatBoostRegressor, 
        feature_names: List[str],
        data: pd.DataFrame, 
        target_name: str,
        id_col: str = "id"
) -> go.Figure:
    """
    Generate a scatter plot of actual vs predicted values with a 45-degree line and hover annotations.

    Parameters:
    - model: Trained CatBoostRegressor
    - feature_names: List of feature column names used in model
    - data: DataFrame including features, target, and ID column
    - target_name: Name of the target variable
    - id_col: Name of the column to use as observation ID

    Returns:
    - plotly.graph_objs._figure.Figure object
    """
    # Predict
    X = data[feature_names]
    y_true = data[target_name]
    y_pred = model.predict(X)
    ids = data[id_col].astype(str).values  # ensure string for annotation

    errors = y_pred - y_true

    # Define diagonal line range
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))

    # Prepare hover text
    hover_text = [
        f"ID: {id_}<br>Actual: {actual:.2f}<br>Predicted: {pred:.2f}<br>Error: {err:+.2f}"
        for id_, actual, pred, err in zip(ids, y_true, y_pred, errors)
    ]

    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        marker=dict(color='rgba(99, 110, 250, 0.6)', size=6),
        text=hover_text,
        hoverinfo='text',
        name='Predicted vs Actual'
    ))

    # Add 45-degree line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='45° Reference Line'
    ))

    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
    )

    return fig


if __name__ == '__main__':
    pass