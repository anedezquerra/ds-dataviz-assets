# -*- coding: utf-8 -*-
import shap
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor

def plot_shap_summary_catboost(model:CatBoostRegressor, data: pd.DataFrame, max_display: int=20):
    # Ensure the model is fitted and shap is available
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    # Prepare data for summary plot
    feature_names = data.columns
    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df_long = shap_df.melt(var_name="Feature", value_name="SHAP Value")

    # Merge feature values to enable coloring (optional)
    value_df = pd.DataFrame(data, columns=feature_names)
    value_df_long = value_df.melt(var_name="Feature", value_name="Feature Value")

    merged = pd.concat([shap_df_long, value_df_long["Feature Value"]], axis=1)

    # Limit to top N features by mean(|shap|)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(max_display).index
    merged = merged[merged["Feature"].isin(top_features)]

    fig = px.strip(
        merged,
        x="SHAP Value",
        y="Feature",
        color="Feature",
        orientation="h",
        title="SHAP Summary Plot",
        stripmode="overlay"
    )
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


if __name__ == '__main__':
    pass

