#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd

import plotly 
import plotly.graph_objects as go

from scipy.stats import f

from typing import List

def plot_t2_chart(
        df: pd.DataFrame, 
        numeric_columns: List[str], 
        alpha: float=0.05,
        UCL_name: str='Upper Control Line (UCL)',
        UCL_line: dict=dict(color='red', dash='dash'),
        T2_line: dict=dict(color='orange', dash='solid'),
        T2_marker: dict=dict(color='blue'),
        outlier_marker: dict=dict(color='red', size=8)
) -> plotly.graph_objs._figure.Figure:
    """
    Description
    -----------
    Generates a Hotelling’s T² control chart using Plotly for multivariate statistical process control (MSPC).
    This chart detects multivariate outliers by comparing each observation’s T² statistic to an upper control limit (UCL)
    derived from the F-distribution. Observations exceeding the UCL are flagged as outliers.

    Input Parameters
    ----------------
    df : pd.DataFrame
        Input dataframe containing at least the numeric variables and an identifier column called 'material_salida'.

    numeric_columns : List[str]
        List of column names in the dataframe that will be used for the multivariate analysis.

    alpha : float, optional (default=0.05)
        Significance level for computing the Upper Control Limit (UCL) using the F-distribution.

    UCL_name : str, optional
        Label for the UCL line in the plot legend.

    UCL_line : dict, optional
        Styling dictionary for the UCL line (color, dash type, etc.).

    T2_line : dict, optional
        Styling dictionary for the line connecting the T² statistics.

    T2_marker : dict, optional
        Styling dictionary for normal point markers.

    outlier_marker : dict, optional
        Styling dictionary for points exceeding the UCL (outliers).

    Results
    -------
    fig : plotly.graph_objs._figure.Figure
        A Plotly Figure object displaying the Hotelling’s T² control chart.

    Example Usage
    -------------
    >>> df = pd.read_csv("process_data.csv")
    >>> numeric_cols = ['feature1', 'feature2', 'feature3']
    >>> fig = plot_t2_chart(df, numeric_cols)
    >>> fig.show()
    """

    df = df.copy()
    X = df[numeric_columns].dropna().values  # Drop rows with NaNs in numeric columns
    material_labels = df['material_salida'].astype(str).values
    
    n, p = X.shape
    X_mean = np.nanmean(X, axis=0)
    S = np.cov(X, rowvar=False)

    # Compute Hotelling's T² for each observation and store the values
    T2 = []
    S_inv = np.linalg.inv(S)
    for x in X:
        diff = x - X_mean
        t2 = diff @ S_inv @ diff.T
        T2.append(t2)

    # Create dataframe with material labels and all needed information
    df_t2 = pd.DataFrame({
        'Index': material_labels,
        'T2': T2,
        'Is_Outlier': [t2 > ((p * (n - 1) * (n + 1) / (n * (n - p))) * f.ppf(1 - alpha, p, n - p)) for t2 in T2]
    })
    
    # Add the original numeric values to the dataframe for hover text
    for i, col in enumerate(numeric_columns):
        df_t2[col] = df[col].values

    # UCL based on F-distribution
    UCL = (p * (n - 1) * (n + 1) / (n * (n - p))) * f.ppf(1 - alpha, p, n - p)

    # Create hover text with all numeric values
    hover_text = []
    for idx, row in df_t2.iterrows():
        text = f"Index: {row['Index']}<br>T²: {row['T2']:.2f}<br>"
        for col in numeric_columns:
            text += f"{col}: {row[col]:.2f}<br>"
        hover_text.append(text)

    # Plot
    fig = go.Figure()

    # Add normal points (below UCL)
    fig.add_trace(go.Scatter(
        x=df_t2['Index'],
        y=df_t2['T2'],
        mode='lines+markers',
        name="T² Statistic",
        marker=dict(
            color=np.where(df_t2['Is_Outlier'], outlier_marker['color'], T2_marker['color']),
            size=np.where(df_t2['Is_Outlier'], outlier_marker.get('size', 8), T2_marker.get('size', 6))
        ),
        line=T2_line,
        hovertext=hover_text,
        hoverinfo="text",
        showlegend=True
    ))

    # Add UCL line
    fig.add_trace(go.Scatter(
        x=df_t2['Index'],
        y=[UCL] * len(df_t2),
        mode='lines',
        name=UCL_name,
        line=UCL_line,
        hoverinfo="none"
    ))

    fig.update_layout(
        title=f"Hotelling's T² Control Chart for the variable combination of: [{', '.join(numeric_columns)}]",
        xaxis_title="Heat ID",
        yaxis_title="T² Statistic",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            type='category',
            tickangle=-90,
            tickmode='array',
            tickvals=df_t2['Index'],
            ticktext=df_t2['Index']
        ),
        hovermode="x unified"
    )

    return fig


if __name__ == '__main__':
    pass