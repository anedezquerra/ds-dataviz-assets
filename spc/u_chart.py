import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_u_chart(df, column, sample_size_col, lower_spec=None, upper_spec=None):
    """
    Description
    -----------
    Generates a U Chart using Plotly to monitor defects per unit with variable sample sizes.
    A U Chart is a type of control chart used in quality control to analyze the number of defects 
    per unit in samples of varying sizes. This implementation visually identifies observations 
    that are out of statistical control based on user-defined specification limits.

    Input Parameters
    ----------------
    df : pandas.DataFrame
        The input dataset containing the data to be analyzed.
    
    column : str
        The name of the column in `df` containing the measurement or quality variable.
    
    sample_size_col : str
        The name of the column in `df` that contains the sample size or number of opportunities 
        per row (denominator for calculating defect rate).
    
    lower_spec : float or None, optional
        The lower specification limit (LSL). If not applicable, set to None.
    
    upper_spec : float or None, optional
        The upper specification limit (USL). If not applicable, set to None.

    Results
    -------
    fig : plotly.graph_objects.Figure
        An interactive Plotly figure showing:
        - Defect rate per unit (`u`) per observation
        - Center Line (u-bar)
        - Control limits (UCL and LCL)
        - Highlighted markers for out-of-control points (above UCL)
        - Custom hover information including original value and `material_salida` info

    Example Usage
    -------------
    >>> fig = plot_u_chart(
    ...     df=data,
    ...     column='t2',
    ...     sample_size_col='sample_size',
    ...     lower_spec=None,
    ...     upper_spec=5.0
    ... )
    >>> fig.show()
    """
    if lower_spec is None and upper_spec is None:
        raise ValueError("At least one of lower_spec or upper_spec must be specified.")

    df = df.copy()

    # Define logic to identify whether a value is defective
    def count_defects(x):
        if lower_spec is not None and x < lower_spec:
            return 1
        if upper_spec is not None and x > upper_spec:
            return 1
        return 0

    # Apply defect logic and compute control chart statistics
    df['defect_count'] = df[column].apply(count_defects)
    df['sample_size'] = df[sample_size_col]
    df['u'] = df['defect_count'] / df['sample_size']

    # Compute overall average defect rate (center line)
    u_bar = df['defect_count'].sum() / df['sample_size'].sum()

    # Compute variable control limits (UCL and LCL)
    df['UCL'] = u_bar + 3 * np.sqrt(u_bar / df['sample_size'])
    df['LCL'] = u_bar - 3 * np.sqrt(u_bar / df['sample_size'])
    df['LCL'] = df['LCL'].clip(lower=0)  # LCL cannot be negative

    # Identify out-of-control points (only those above UCL)
    df['out_of_control'] = df['u'] > df['UCL']

    # Create custom hover text with detailed information
    df['hover_text'] = (
        "Index: " + df.index.astype(str) +
        "<br>" + column + ": " + df[column].astype(str) +
        "<br>material_salida: " + df['material_salida'].astype(str)
    )

    # Initialize Plotly figure
    fig = go.Figure()

    # In-control points with blue markers and connecting lines
    fig.add_trace(go.Scatter(
        x=df.index[~df['out_of_control']],
        y=df['u'][~df['out_of_control']],
        mode='markers+lines',
        name='u (Defects per Unit)',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        text=df['hover_text'][~df['out_of_control']],
        hoverinfo='text'
    ))

    # Out-of-control points shown in red with open circle markers
    fig.add_trace(go.Scatter(
        x=df.index[df['out_of_control']],
        y=df['u'][df['out_of_control']],
        mode='markers',
        name='Out of Control',
        marker=dict(color='red', size=10, symbol='circle-open'),
        text=df['hover_text'][df['out_of_control']],
        hoverinfo='text'
    ))

    # Add center line (u-bar)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=[u_bar] * len(df),
        mode='lines',
        name='Center Line (ū)',
        line=dict(color='green', dash='dash')
    ))

    # Add upper and lower control limits
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['UCL'],
        mode='lines',
        name='UCL',
        line=dict(color='red', dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['LCL'],
        mode='lines',
        name='LCL',
        line=dict(color='red', dash='dot')
    ))

    # Configure layout: labels, template, and external bottom-centered legend
    fig.update_layout(
        title=f"U Chart for {column}",
        xaxis_title="Observation",
        yaxis_title="Defects per Unit",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    return fig

if __name__ == '__main__':
    pass