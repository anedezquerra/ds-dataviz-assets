#-*- coding:utf-8 -*-

# from catboost import CatBoostRegressor, Pool

import plotly.express as px
import pandas as pd

def plot_catboost_global_explanations(model, feature_names=None, top_n=20):
    """
    Plots global feature importance from a trained CatBoostRegressor using Plotly.

    Parameters:
    - model: Trained CatBoostRegressor
    - feature_names: List of feature names (optional, required if model was trained on numpy arrays)
    - top_n: Number of top features to display (default: 20)
    """
    importances = model.get_feature_importance()
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Global Feature Importance - CatBoost Regressor',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
    )

    fig.update_layout(yaxis=dict(autorange='reversed'), template='plotly_white')
    
    return fig


if __name__ == '__main__':
    pass