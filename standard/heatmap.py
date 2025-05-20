# -*-coding:utf-8 -*-

import pandas as pd
import plotly.express as px

def plot_heatmap(
        df, 
        columns_to_include,
        methods: str
    ):
    """
    Generates a correlation heatmap using Plotly for the specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_include (list): List of column names to include in the heatmap.

    Returns:
    Plotly Figure object.
    """
    # Validate input columns
    valid_columns = [col for col in columns_to_include if col in df.columns]
    if not valid_columns:
        raise ValueError("None of the selected columns are present in the DataFrame.")

    # Subset and compute correlation matrix
    corr_matrix = df[valid_columns].corr(method=methods)

    # Generate heatmap with Plotly
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=f"Correlation Heatmap for [{', '.join(valid_columns)}]",
        labels=dict(color="Correlation")
    )
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables",
        xaxis_showgrid=False,
        yaxis_showgrid=False
    )
    return fig


if __name__ == '__main__':
    pass
