import plotly.express as px

def plot_histogram(df, numeric_col, categorical_col=None, bins=30):
    """
    Plots a histogram using Plotly.

    Parameters:
    - df: pandas DataFrame
    - numeric_col: str, name of the numerical column (e.g., 'n_EAF_acero')
    - categorical_col: str or None, name of the categorical column to facet by (optional)
    - bins: int, number of histogram bins (default = 30)
    """
    df = df.copy()

    if categorical_col:
        df[categorical_col] = df[categorical_col].astype(str)  # ensure categorical for grouping
        fig = px.histogram(
            df,
            x=numeric_col,
            color=categorical_col,
            nbins=bins,
            barmode="overlay",  # or "group" if you prefer side-by-side bars
            title=f"Histogram of '{numeric_col}' grouped by '{categorical_col}'"
        )
    else:
        fig = px.histogram(
            df,
            x=numeric_col,
            nbins=bins,
            title=f"Histogram of '{numeric_col}'"
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title=numeric_col,
        yaxis_title="Count"
    )

    return fig

if __name__ == '__main__':
    pass
