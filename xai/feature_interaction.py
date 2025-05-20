# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor

def plot_feature_interactions_catboost(model: CatBoostRegressor, data: pd.DataFrame, top_n: int = 20):
    """
    Generates a feature interaction chart from a CatBoostRegressor model using Plotly,
    with color-coded categories based on interaction strength.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame used to train the model
    - top_n: Number of top interactions to plot

    Returns:
    - plotly.graph_objects.Figure
    """
    feature_names = data.columns.tolist()

    # Get interaction values from CatBoost
    interactions = model.get_feature_importance(
        type='Interaction',
        prettified=False
    )

    # Convert to DataFrame
    interaction_df = pd.DataFrame(interactions, columns=['Feature 1 Index', 'Feature 2 Index', 'Strength'])

    # Ensure indices are integers
    interaction_df["Feature 1 Index"] = interaction_df["Feature 1 Index"].astype(int)
    interaction_df["Feature 2 Index"] = interaction_df["Feature 2 Index"].astype(int)

    # Map indices to feature names
    interaction_df["Feature 1"] = interaction_df["Feature 1 Index"].apply(lambda i: feature_names[i])
    interaction_df["Feature 2"] = interaction_df["Feature 2 Index"].apply(lambda i: feature_names[i])

    # Sort by strength and get top N
    interaction_df = interaction_df.sort_values(by='Strength', ascending=False).head(top_n)

    # Categorize interaction strength
    strengths = interaction_df['Strength']
    if strengths.nunique() == 1:
        # Flat case: manual bins
        min_s = strengths.min()
        max_s = strengths.max()
        low_thresh = min_s + (max_s - min_s) / 3
        high_thresh = min_s + 2 * (max_s - min_s) / 3
    else:
        # Use quantiles
        low_thresh = strengths.quantile(0.33)
        high_thresh = strengths.quantile(0.66)

    def categorize_strength(val):
        if val <= low_thresh:
            return 'Low (Orange)'
        elif val <= high_thresh:
            return 'Medium (Green)'
        else:
            return 'High (Red)'

    interaction_df['Strength Category'] = interaction_df['Strength'].apply(categorize_strength)

    color_map = {
        'Low (Orange)': 'orange',
        'Medium (Green)': 'green',
        'High (Red)': 'red'
    }

    # Plot
    fig = px.scatter(
        interaction_df,
        x="Feature 1",
        y="Feature 2",
        size="Strength",
        color="Strength Category",
        color_discrete_map=color_map,
        hover_data=["Strength"],
        title=f"Top {top_n} Feature Interaction Strengths."
    )

    # Legend positioning
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title_text="Interaction Strength Category",
            traceorder="normal"
        ),
        legend_tracegroupgap=5,
        template='plotly_white'
    )

    return fig
