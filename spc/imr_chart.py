# -*- coding: utf-8 -*-

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd

def plot_imr_chart(
        df: pd.DataFrame, 
        value_column: str=None,
        grouping_column: str=None
    ):
    """
    Interactive I-MR chart with synchronized zoom/hover using Plotly.
    - df: pandas DataFrame
    - value_column: name of the measurement value_column
    """
    # Data preparation
    data_series = df[value_column].dropna()
    data = data_series.values
    indices = data_series.index
    
    if len(data) < 2:
        raise ValueError("At least 2 data points required for Moving Range calculation")

    # Calculate moving ranges
    moving_ranges = np.abs(np.diff(data))
    
    # Get material_salida values
    material_salida_individuals = [df.loc[idx, grouping_column] for idx in indices]
    material_salida_mr = [
        f"{df.loc[indices[i], grouping_column]}, {df.loc[indices[i+1], grouping_column]}" 
        for i in range(len(indices)-1)
    ]

    # Control limit calculations
    X_bar = np.mean(data)
    MR_bar = np.mean(moving_ranges)
    
    # Constants for n=2 (moving range between consecutive points)
    E2 = 2.659  # For individuals chart
    D3 = 0       # For MR chart
    D4 = 3.267   # For MR chart

    # Control limits
    UCL_x = X_bar + E2 * MR_bar
    LCL_x = X_bar - E2 * MR_bar
    UCL_mr = D4 * MR_bar
    LCL_mr = D3 * MR_bar

    # Create figure with synchronized axes
    fig = make_subplots(
        rows=2, cols=1,
        #shared_x=True,  # Initial synchronization
        vertical_spacing=0.15,
        subplot_titles=(
            'Individuals Chart',
            'Moving Range Chart'
        )
    )

    # Custom hover templates
    hover_template_individual = (
        "<b>Point %{x}</b><br>"
        "Value: %{y:.2f}<br>"
        "{grouping_column}: %{customdata}<extra></extra>"
    )
    
    hover_template_mr = (
        "<b>Points %{x}-%{text}</b><br>"
        "Range: %{y:.2f}<br>"
        "{grouping_column}: %{customdata}<extra></extra>"
    )

    # Individuals Chart
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(data)),
            y=data,
            mode='lines+markers',
            name='Individual',
            customdata=material_salida_individuals,
            hovertemplate=hover_template_individual,
            marker=dict(color='#1f77b4'),
            line=dict(width=1)
        ),
        row=1, col=1
    )

    # Moving Range Chart
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(moving_ranges)),
            y=moving_ranges,
            mode='lines+markers',
            name='Moving Range',
            customdata=material_salida_mr,
            hovertemplate=hover_template_mr,
            text=[str(i+1) for i in range(len(moving_ranges))],  # Shows end point of range
            marker=dict(color='#ff7f0e'),
            line=dict(width=1)
        ),
        row=2, col=1
    )

    # Add control limits
    for row, (UCL, LCL, CL, data_points) in enumerate(zip(
        [UCL_x, UCL_mr], [LCL_x, LCL_mr], [X_bar, MR_bar], [data, moving_ranges]
    ), start=1):
        fig.add_hline(
            y=UCL, line=dict(color='red', dash='dash'),
            row=row, col=1,
            annotation_text=f'UCL: {UCL:.2f}',
            annotation_position="top right"
        )
        fig.add_hline(
            y=LCL, line=dict(color='red', dash='dash'),
            row=row, col=1,
            annotation_text=f'LCL: {LCL:.2f}',
            annotation_position="bottom right"
        )
        fig.add_hline(
            y=CL, line=dict(color='black'),
            row=row, col=1,
            annotation_text=f'CL: {CL:.2f}',
            annotation_position="top left"
        )

    # Update layout
    fig.update_layout(
        height=1200,
        title_text=f"Interactive I-MR Control Charts: Analysis column: {value_column} - Grouping column: {grouping_column}",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        margin=dict(t=100),
        xaxis=dict(
            showticklabels=False,
            matches='x2'  # Link to second subplot's x-axis
        ),
        xaxis2=dict(
            title='Observation Number',
            matches='x'  # Link to first subplot's x-axis
        )
    )

    # Axis labels
    fig.update_yaxes(title_text="Individual Values", row=1, col=1)
    fig.update_yaxes(title_text="Moving Range", row=2, col=1)
    fig.update_xaxes(title_text="Observation Number", row=2, col=1)

    return fig

if __name__ == '__main__':
    pass