#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import sqrt

def plot_ewma_chart(
        df: pd.DataFrame, 
        value_column: str=None, 
        lambda_value: float=0.2, 
        l_value: int=3
):
    """
    Generates an EWMA control chart using Plotly.

    Parameters:
    - df: pandas DataFrame
    - value_column: str, name of the value_column to chart
    - lambda_value: float, smoothing constant (between 0 and 1)
    - l_value: float, control limit width (usually 3 for 3-sigma)
    """
    df = df.copy()
    x = df[value_column].values
    n = len(x)

    mu = np.mean(x)
    sigma = np.std(x, ddof=1)

    # Initialize EWMA
    ewma = np.zeros(n)
    ewma[0] = mu  # Start with process mean

    for i in range(1, n):
        ewma[i] = lambda_value * x[i] + (1 - lambda_value) * ewma[i - 1]

    # Control limits
    std_dev = [sqrt(lambda_value / (2 - lambda_value)) * sigma * sqrt(1 - (1 - lambda_value)**(2 * (i + 1))) for i in range(n)]
    ucl = mu + l_value * np.array(std_dev)
    lcl = mu - l_value * np.array(std_dev)

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=ewma,
        mode='lines+markers',
        name='EWMA',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[mu] * n,
        mode='lines',
        name='Center Line (μ)',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=ucl,
        mode='lines',
        name='UCL',
        line=dict(color='red', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=lcl,
        mode='lines',
        name='LCL',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title=f"Exponentially Weighted Moving Average Chart (EWMA Chart) for '{value_column}'",
        xaxis_title="Observation",
        yaxis_title=value_column,
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':
    pass
