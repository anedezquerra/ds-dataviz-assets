import pandas as pd
import plotly.graph_objects as go
import numpy as np

def plot_p_chart(df, column, subgroup_size=10, lower_spec=None, upper_spec=None):
    """
    Creates a P Chart (Proportion Defective chart) using Plotly.

    Parameters:
    - df: pandas DataFrame containing the data
    - column: str, column name to analyze
    - subgroup_size: int, number of samples per subgroup
    - lower_spec: float or None, lower specification limit
    - upper_spec: float or None, upper specification limit
    """
    # Define defect condition
    if lower_spec is None and upper_spec is None:
        raise ValueError("At least one of lower_spec or upper_spec must be defined.")

    def is_defective(x):
        if lower_spec is not None and x < lower_spec:
            return 1
        if upper_spec is not None and x > upper_spec:
            return 1
        return 0

    df = df.copy()
    df['defective'] = df[column].apply(is_defective)

    # Create subgroups
    df['subgroup'] = df.index // subgroup_size
    grouped = df.groupby('subgroup')['defective']
    subgroup_counts = grouped.count()
    subgroup_defectives = grouped.sum()
    p = subgroup_defectives / subgroup_counts

    # Center line (CL)
    p_bar = p.mean()

    # Control limits
    UCL = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / subgroup_size)
    LCL = p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / subgroup_size)
    LCL = max(0, LCL)  # LCL can't be negative

    # Plotly chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p.index,
        y=p,
        mode='lines+markers',
        name='Proportion Defective',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=p.index,
        y=[p_bar]*len(p),
        mode='lines',
        name='Center Line (CL)',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=p.index,
        y=[UCL]*len(p),
        mode='lines',
        name='UCL',
        line=dict(color='red', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=p.index,
        y=[LCL]*len(p),
        mode='lines',
        name='LCL',
        line=dict(color='red', dash='dot')
    ))

    fig.update_layout(
        title=f"P Chart for {column}",
        xaxis_title="Subgroup",
        yaxis_title="Proportion Defective",
        yaxis=dict(range=[0, max(UCL*1.1, 0.05)]),
        template="plotly_white"
    )

    return fig

if __name__ == '__main__':
    pass