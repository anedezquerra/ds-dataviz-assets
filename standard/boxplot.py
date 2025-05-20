#-*- coding:utf-8 -*-

import plotly.express as px

def plot_boxplot(data, numeric_col, categorical_col=None):
    """
    Plots a Plotly boxplot. If categorical_col is provided, it's used as the categorical axis.
    Otherwise, a simple boxplot of numeric_col is shown.

    Parameters:
    - df: pandas DataFrame
    - numeric_col: str, name of the numerical column (e.g., 'n_EAF_acero')
    - categorical_col: str or None, name of the categorical column (e.g., 'tipo_acero')
    """
    df = data.copy()

    if categorical_col:
        df[categorical_col] = df[categorical_col].astype(str)  # ensure it's treated as categorical
        fig = px.box(
            df,
            x=categorical_col,
            y=numeric_col,
            color=categorical_col,
            points="all",
            title=f"Boxplot of '{numeric_col}' by '{categorical_col}'"
        )
        fig.update_layout(showlegend=False)
    else:
        fig = px.box(
            df,
            y=numeric_col,
            points="all",
            title=f"Boxplot of '{numeric_col}'"
        )

    fig.update_layout(
        template="plotly_white",
        yaxis_title=numeric_col,
        xaxis_title=categorical_col if categorical_col else ""
    )

    return fig


if __name__ == '__main__':
    pass