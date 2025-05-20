#-*- coding: utf-8 -*-

import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_shap_decision_catboost(model, data: pd.DataFrame, indices=[0], feature_names=None):
    """
    Generate a SHAP Decision Plot using Plotly for CatBoost Regressor.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame with input features
    - indices: List of row indices to plot
    - feature_names: Optional list of feature names

    Returns:
    - fig: plotly.graph_objs._figure.Figure
    """
    if feature_names is None:
        feature_names = data.columns.tolist()

    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    base_value = shap_values.base_values[0]

    fig = go.Figure()
    for idx in indices:
        row_values = shap_values[idx].values
        cumulative = np.cumsum(np.insert(row_values, 0, 0)) + base_value
        features = feature_names

        fig.add_trace(go.Scatter(
            x=np.arange(len(features) + 1),
            y=cumulative,
            mode="lines+markers",
            name=f"Index {idx}",
            hoverinfo="text",
            text=["Base Value"] + [f"{features[i]}: {row_values[i]:+.3f}" for i in range(len(features))],
        ))

    # Format x-axis with feature names + base value
    x_labels = ["Base Value"] + features
    fig.update_layout(
        title="SHAP Decision Plot",
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(x_labels)),
            ticktext=x_labels,
            title="Features (in order)"
        ),
        yaxis_title="Model Output",
        template="plotly_white",
        showlegend=True
    )

    return fig


if __name__ == '__main__':
    pass