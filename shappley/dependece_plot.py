import shap
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor

def plot_shap_dependence_catboost(
        model: CatBoostRegressor, 
        data: pd.DataFrame, 
        feature: str, 
        interaction_index: str = None
    ):
    """
    Generates a SHAP dependence plot using Plotly for a CatBoost Regressor model.

    Parameters:
    - model: Trained CatBoostRegressor model
    - data: pandas DataFrame with feature data
    - feature: str, the feature to plot SHAP dependence for
    - interaction_index: str, optional feature to color by (default: auto-detected by SHAP)

    Returns:
    - fig: Plotly Figure object
    """

    # Ensure SHAP is compatible
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=data.columns)
    feature_values = data[feature]

    # Choose interaction feature (default SHAP logic)
    if interaction_index is None:
        interaction_index = shap_values[:, feature].values.argmax(axis=0)
        interaction_index = data.columns[interaction_index] if isinstance(interaction_index, int) else feature

    color_values = data[interaction_index] if interaction_index in data.columns else None

    fig = px.scatter(
        x=feature_values,
        y=shap_df[feature],
        color=color_values,
        labels={
            "x": feature,
            "y": f"SHAP value for {feature}",
            "color": interaction_index
        },
        title=f"SHAP Dependence Plot for variable: '{feature} indexed by: {interaction_index}'",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(template="plotly_white")

    return fig

if __name__ == '__main__':
    pass