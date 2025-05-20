import shap
import pandas as pd
import plotly.graph_objects as go

def plot_shap_waterfall_catboost(model, data: pd.DataFrame, row_index: int, feature_names=None):
    """
    Generate a SHAP Waterfall Plot for a specific instance using Plotly.

    Parameters:
    - model: Trained CatBoostRegressor
    - data: pandas DataFrame of input features
    - row_index: Integer row_index of the row to explain
    - feature_names: Optional list of feature names
    
    Returns:
    - fig: plotly.graph_objs._figure.Figure
    """

    # Ensure feature names
    if feature_names is None:
        feature_names = data.columns.tolist()

    # Compute SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    # Get base value, SHAP values, and data row
    row_shap_values = shap_values[row_index].values
    base_value = shap_values.base_values[row_index]
    predicted_value = base_value + row_shap_values.sum()
    row_data = data.iloc[row_index]

    # Prepare waterfall data
    sorted_idx = abs(row_shap_values).argsort()[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_shap = row_shap_values[sorted_idx]
    sorted_values = row_data.values[sorted_idx]

    fig = go.Figure()
    
    cum_base = base_value
    for i, (feat, shap_val, val) in enumerate(zip(sorted_features, sorted_shap, sorted_values)):
        fig.add_trace(go.Waterfall(
            name=feat,
            orientation='v',
            measure=["relative"],
            x=[feat],
            y=[shap_val],
            text=[f"{feat} = {val:.2f}"],
            textposition="outside",
            connector={"line": {"color": "rgba(63, 63, 63, 0.7)"}},
        ))

    fig.update_layout(
        title=f"SHAP Waterfall Plot (row_index {row_index})",
        showlegend=False,
        waterfallgap=0.5,
        yaxis_title="Model Output",
        template="plotly_white",
    )

    # Add base and prediction as annotations
    fig.add_annotation(
        text=f"Base Value: {base_value:.2f}<br>Prediction: {predicted_value:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    return fig


if __name__ == '__main__':
    pass