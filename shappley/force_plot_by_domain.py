# -*- coding:utf-8 -*-

import shap
import plotly
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

from catboost import CatBoostRegressor

def plot_shap_force_catboost_by_domain(
        model: CatBoostRegressor,
        data: pd.DataFrame, 
        row_index: int=0, 
        feature_domain_map=None, 
        domain_colors=None
) -> plotly.graph_objs._figure.Figure:
    """
    Approximates a SHAP force plot using a horizontal bar chart in Plotly, with colors based on feature domains.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame
    - row_index: int, the row index to explain
    - feature_domain_map: dict, mapping of feature name to domain
    - domain_colors: dict, mapping of domain to color (optional)

    Returns:
    - Plotly Figure object
    """

    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    base_value = shap_values.base_values[row_index]
    row_shap_values = shap_values.values[row_index]
    row_features = data.iloc[row_index]

    df_contrib = pd.DataFrame({
        'Feature': row_features.index,
        'SHAP Value': row_shap_values,
        'Feature Value': row_features.values
    }).sort_values('SHAP Value', key=abs, ascending=False)

    df_contrib['Cumulative'] = base_value + df_contrib['SHAP Value'].cumsum()

    # Assign domains
    if feature_domain_map:
        df_contrib['Domain'] = df_contrib['Feature'].map(feature_domain_map)
    else:
        df_contrib['Domain'] = 'Unknown'

    # Assign colors
    if not domain_colors:
        # Assign default palette
        unique_domains = df_contrib['Domain'].unique()
        default_colors = px.colors.qualitative.Set3
        domain_colors = {domain: default_colors[i % len(default_colors)] for i, domain in enumerate(unique_domains)}

    df_contrib['Color'] = df_contrib['Domain'].map(domain_colors)

    # Plot
    fig = go.Figure()
    current_position = base_value

    for _, row in df_contrib.iterrows():
        fig.add_trace(go.Bar(
            y=["Prediction"],
            x=[row['SHAP Value']],
            base=current_position,
            orientation='h',
            name=f"{row['Feature']} = {row['Feature Value']:.2f}",
            marker_color=row['Color'],
            hovertemplate=f"{row['Feature']}<br>Value: {row['Feature Value']:.2f}<br>SHAP: {row['SHAP Value']:.3f}<br>Domain: {row['Domain']}<extra></extra>"
        ))
        current_position += row['SHAP Value']

    # Add predicted value marker
    prediction = base_value + row_shap_values.sum()
    fig.add_shape(
        type="line",
        x0=prediction, x1=prediction, y0=-0.5, y1=0.5,
        line=dict(color="black", width=2, dash="dash"),
        name="Prediction"
    )

    fig.update_layout(
        title=f"SHAP Force Plot (Domain Colored) for Row {row_index}",
        barmode='stack',
        xaxis_title="Prediction Value",
        yaxis=dict(showticklabels=False),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5)
    )

    return fig

if __name__ == '__main__':
    pass