#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from catboost import Pool
import plotly.graph_objects as go

def get_optimal_grid_points(n_rows: int):
    if n_rows < 200:
        return 10
    elif n_rows < 500:
        return 15
    elif n_rows < 1000:
        return 20
    elif n_rows < 5000:
        return 25
    else:
        return 30

def plot_pdp_catboost(model, data: pd.DataFrame, features: list, grid_points: int = None):
    """
    Computes and plots combined Partial Dependence for multiple numeric features in a single plot.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame of input features
    - features: List of feature names (strings)
    - grid_points: Optional int, number of grid points per feature

    Returns:
    - plotly.graph_objects.Figure with multiple PDP traces
    """
    if grid_points is None:
        grid_points = get_optimal_grid_points(len(data))

    fig = go.Figure()

    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Feature '{feature}' not in DataFrame columns.")

        grid = np.linspace(data[feature].min(), data[feature].max(), grid_points)
        predictions = []

        for val in grid:
            temp_data = data.copy()
            temp_data[feature] = val
            pool = Pool(temp_data)
            preds = model.predict(pool)
            predictions.append(preds.mean())

        fig.add_trace(go.Scatter(
            x=grid,
            y=predictions,
            mode='lines+markers',
            name=feature
        ))

    fig.update_layout(
        title=f"Combined Partial Dependence Plots of variables [{', '.join(features)}]",
        xaxis_title='Feature Value',
        yaxis_title='Average Prediction',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,  # Push legend outside plot
            xanchor="center",
            x=0.5,
            traceorder="normal",
            title_text='Features',
            itemsizing='trace',
            valign="middle",
            font=dict(size=12),
            tracegroupgap=5,
        ),
        legend_tracegroupgap=0,
        legend_itemclick="toggleothers"
    )

    # Set the number of columns for the legend using legendgroup titles
    fig.update_layout(legend=dict(traceorder='normal', orientation="h", x=0.5, xanchor="center", y=-0.3, yanchor="bottom", itemsizing='constant'), legend_tracegroupgap=0)


    return fig



if __name__ == '__main__':
    pass
