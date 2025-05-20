# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor

def plot_feature_importance_catboost(
        model: CatBoostRegressor, 
        data: pd.DataFrame, 
        top_n: int = 20
):
    """
    Generates a horizontal bar chart showing feature importances from a trained CatBoostRegressor model.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame used for training (to extract feature names)
    - top_n: Number of top features to display

    Returns:
    - plotly.graph_objects.Figure
    """
    feature_names = data.columns.tolist()

    # Get importances
    importances = model.get_feature_importance(type='FeatureImportance')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort and filter top N
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    # Plot
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"Top {top_n}th Feature Importance",
        labels={'Importance': 'Feature Importance', 'Feature': 'Features'},
        color='Importance',
        color_continuous_scale='RdBu'
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )

    return fig


if __name__ == '__main__':
    pass