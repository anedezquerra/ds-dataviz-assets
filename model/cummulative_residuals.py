# -*- coding:utf-8 -*-

from typing import List
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

def plot_cumulative_error_distribution(
        model: CatBoostRegressor,
        feature_names: List[str],
        data: pd.DataFrame,
        target_name: str
) -> go.Figure:
    """
    Generate a Cumulative Error Distribution Plot (ECDF of absolute residuals)
    using Plotly for Dash dashboards.

    Parameters:
    - model: Trained CatBoostRegressor
    - feature_names: List of features used for training
    - data: DataFrame with features and target
    - target_name: Column name of the target variable

    Returns:
    - plotly.graph_objs._figure.Figure object
    """
    # Predict and calculate residuals
    X = data[feature_names]
    y_true = data[target_name]
    y_pred = model.predict(X)
    residuals = np.abs(y_true - y_pred)

    # Sort residuals for ECDF
    sorted_residuals = np.sort(residuals)
    ecdf = np.arange(1, len(sorted_residuals) + 1) / len(sorted_residuals)

    fig = go.Figure()

    # Plot ECDF
    fig.add_trace(go.Scatter(
        x=sorted_residuals,
        y=ecdf,
        mode='lines+markers',
        marker=dict(color='rgba(99, 110, 250, 0.6)', size=5),
        name='Cumulative Error Distribution',
        hovertemplate='|Residual|: %{x:.3f}<br>Cumulative %: %{y:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title='Cumulative Error Distribution of Residuals',
        xaxis_title='Absolute Residual',
        yaxis_title='Cumulative Probability',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )

    return fig

if __name__ == '__main__':
    pass