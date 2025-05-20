import pandas as pd
import plotly.graph_objects as go
import numpy as np

def plot_c_chart(df, column, lower_spec=None, upper_spec=None):
    """
    Creates a C Chart (count of defects per unit) using Plotly.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: str, the column to analyze
    - lower_spec: float or None, lower specification limit for defects
    - upper_spec: float or None, upper specification limit for defects
    """
    if lower_spec is None and upper_spec is None:
        raise ValueError("You must define at least one specification limit.")

    # Define what is considered a defect
    def count_defects(x):
        defect_count = 0
        if lower_spec is not None and x < lower_spec:
            defect_count += 1
        if upper_spec is not None and x > upper_spec:
            defect_count += 1
        return defect_count

    df = df.copy()
    df['defect_count'] = df[column].apply(count_defects)

    # Center line (CL)
    c_bar = df['defect_count'].mean()

    # Control limits
    UCL = c_bar + 3 * np.sqrt(c_bar)
    LCL = max(0, c_bar - 3 * np.sqrt(c_bar))

    # Plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['defect_count'],
        mode='lines+markers',
        name='Defects per Unit',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[c_bar]*len(df),
        mode='lines',
        name='Center Line (CL)',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[UCL]*len(df),
        mode='lines',
        name='UCL',
        line=dict(color='red', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=[LCL]*len(df),
        mode='lines',
        name='LCL',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title=f"C Chart for {column}",
        xaxis_title="Observation",
        yaxis_title="Defects per Unit",
        yaxis=dict(range=[0, max(UCL * 1.1, df['defect_count'].max() + 1)]),
        template="plotly_white"
    )

    return fig


if __name__ == '__main__':
    pass