#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go

def plot_ecdf(
        data: pd.DataFrame,
        numeric_col: str=None
) -> plotly.graph_objs._figure.Figure:
    """
    Plots an ECDF (Empirical Cumulative Distribution Function) using Plotly.

    Parameters:
    - df: pandas DataFrame
    - numeric_col: str, the numerical column to plot
    """
    df = data.copy()
    df = np.sort(df[numeric_col].dropna())
    n = data.size
    y = np.arange(1, n + 1) / n  # ECDF values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data,
        y=y,
        mode='lines+markers',
        name=f'ECDF of {numeric_col}'
    ))

    fig.update_layout(
        title=f"ECDF Plot of '{numeric_col}'",
        xaxis_title=numeric_col,
        yaxis_title='ECDF',
        template='plotly_white',
        yaxis=dict(range=[0, 1])
    )

    return fig

if __name__ == '__main__':
    pass
