# -*- coding:utf-8 -*-

from typing import List
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import probplot
from catboost import CatBoostRegressor

def plot_qq_residuals(
        model: CatBoostRegressor,
        feature_names: List[str],
        data: pd.DataFrame,
        target_name: str
) -> go.Figure:
    """
    Generate a Q-Q plot of residuals using Plotly for Dash.

    Parameters:
    - model: Trained CatBoostRegressor
    - feature_names: List of feature column names used for training
    - data: DataFrame including features and target
    - target_name: Name of the target variable

    Returns:
    - plotly.graph_objs._figure.Figure object
    """
    # Predict and calculate residuals
    X = data[feature_names]
    y_true = data[target_name]
    y_pred = model.predict(X)
    residuals = y_true - y_pred

    # Generate Q-Q data using scipy
    qq = probplot(residuals, dist="norm", plot=None)
    theoretical_quants = qq[0][0]
    sample_quants = qq[0][1]

    fig = go.Figure()

    # Scatter plot of theoretical vs sample quantiles
    fig.add_trace(go.Scatter(
        x=theoretical_quants,
        y=sample_quants,
        mode='markers',
        marker=dict(color='rgba(0, 123, 255, 0.6)', size=6),
        name='Q-Q Points',
        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
    ))

    # Add 45-degree reference line
    min_val = min(min(theoretical_quants), min(sample_quants))
    max_val = max(max(theoretical_quants), max(sample_quants))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='45° Reference Line'
    ))

    fig.update_layout(
        title='Q-Q Plot of Residuals',
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles (Residuals)',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )

    return fig

if __name__ == '__main__':
    pass