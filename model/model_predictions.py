#-*- coding: utf-8 -*-


from typing import List

import plotly
import pandas as pd

import plotly.graph_objects as go


def plot_model_vs_actual_timeseries_with_hover(
    model,
    data: pd.DataFrame,
    feature_names: List[str],
    target_name: str,
    id_col: str
) -> plotly.graph_objs._figure.Figure:
    """
    Plots actual vs predicted values from a CatBoost Regressor in a time-series style chart,
    using the observation ID (as string) on the x-axis and showing all features in hover.

    Parameters:
    - model: Trained CatBoost Regressor
    - df: pandas DataFrame, includes all df (features + target + ID)
    - feature_names: list of str, names of features used by the model
    - target_name: str, name of the target variable
    - id_col: str, name of the observation ID column (must be sortable)

    Returns:
    - fig: plotly.graph_objs._figure.Figure
    """
    # Ensure ID column is string and sort the DataFrame
    df = data.copy()
    df[id_col] = df[id_col].astype(str)
    df.sort_values(by=id_col, inplace=True)

    X = df[feature_names]
    y_true = df[target_name]
    y_pred = model.predict(X)
    x_vals = df[id_col]

    # Create custom hover text
    hover_texts = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = f"<b>{id_col}:</b> {row[id_col]}<br><b>Actual:</b> {row[target_name]}<br><b>Predicted:</b> {y_pred[i]:.2f}"
        for feat in feature_names:
            text += f"<br><b>{feat}:</b> {row[feat]}"
        hover_texts.append(text)

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_true,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue'),
        marker=dict(size=4),
        hoverinfo='text',
        hovertext=hover_texts
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', dash='dash'),
        marker=dict(size=4),
        hoverinfo='text',
        hovertext=hover_texts
    ))

    fig.update_layout(
        title='Model Predictions vs. Actual Values',
        xaxis_title='',
        yaxis_title=target_name,
        xaxis_tickangle=90,
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
        template='plotly_white',
        height=500
    )

    return fig


if __name__ == '__main__':
    pass