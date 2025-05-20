import pandas as pd
import plotly.graph_objects as go
import numpy as np

def plot_np_chart(df, column, subgroup_size=10, lower_spec=None, upper_spec=None):
    """
    Creates an NP Chart (Number of Defectives) using Plotly.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: str, column name to analyze
    - subgroup_size: int, number of samples per subgroup (assumed constant)
    - lower_spec: float or None, lower specification limit
    - upper_spec: float or None, upper specification limit
    """
    if lower_spec is None and upper_spec is None:
        raise ValueError("At least one of lower_spec or upper_spec must be specified.")

    def is_defective(x):
        if lower_spec is not None and x < lower_spec:
            return 1
        if upper_spec is not None and x > upper_spec:
            return 1
        return 0

    df = df.copy()
    df['defective'] = df[column].apply(is_defective)
    df['subgroup'] = df.index // subgroup_size

    grouped = df.groupby('subgroup')['defective']
    np_values = grouped.sum()  # number of defectives per subgroup

    # Center line
    np_bar = np_values.mean()
    p_bar = np_bar / subgroup_size

    # Control limits
    UCL = subgroup_size * (p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / subgroup_size))
    LCL = subgroup_size * (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / subgroup_size))
    LCL = max(0, LCL)

    # Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np_values.index,
        y=np_values,
        mode='lines+markers',
        name='Number Defective',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=np_values.index,
        y=[np_bar]*len(np_values),
        mode='lines',
        name='Center Line (CL)',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=np_values.index,
        y=[UCL]*len(np_values),
        mode='lines',
        name='UCL',
        line=dict(color='red', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=np_values.index,
        y=[LCL]*len(np_values),
        mode='lines',
        name='LCL',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title=f"NP Chart for {column}",
        xaxis_title="Subgroup",
        yaxis_title="Number of Defectives",
        yaxis=dict(range=[0, max(UCL*1.1, np_values.max()+1)]),
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':
    pass