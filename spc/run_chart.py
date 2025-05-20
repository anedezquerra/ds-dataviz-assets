#-*- coding:utf-8 -*-

import pandas as pd
import plotly.graph_objects as go

def plot_run_chart(df, column, time_col=None, center_line='mean'):
    """
    Generates a Run Chart using Plotly.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: str, the column to plot
    - time_col: str or None, column to use as time index (optional)
    - center_line: 'mean' or 'median'
    """
    df = df.copy()

    if time_col:
        df = df.sort_values(by=time_col)
        x_vals = df[time_col]
    else:
        x_vals = df.index

    y_vals = df[column]

    # Determine center line
    if center_line == 'mean':
        cl_value = y_vals.mean()
    elif center_line == 'median':
        cl_value = y_vals.median()
    else:
        raise ValueError("center_line must be 'mean' or 'median'")

    # Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        name=column,
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=[cl_value] * len(df),
        mode='lines',
        name=f'Center Line ({center_line})',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title=f"Run Chart for '{column} for statistic: {center_line}'",
        xaxis_title=time_col if time_col else "Observation",
        yaxis_title=column,
        template="plotly_white"
    )

    return fig


if __name__ == '__main__':
    pass