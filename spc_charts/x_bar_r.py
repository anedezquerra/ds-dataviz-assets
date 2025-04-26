#-*- coding: utf-8 -*-


# --- Built-in packages
import typing
import traceback

# --- Third party packages
import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Local packages



def xbar_r_chart_plotly(
        df: pd.DataFrame=None, 
        value_column: str=None, 
        grouping_column: str=None,
        subgroup_size: int=5
) -> plotly.graph_objs._figure.Figure:
    """
    """
    data_series = df[value_column].dropna()
    data = data_series.values
    indices = data_series.index
    
    k = len(data) // subgroup_size
    if k == 0:
        raise ValueError("Subgroup size is larger than the number of non-NA data points.")
    
    groups = data[:k*subgroup_size].reshape((k, subgroup_size))
    indices_subgroups = [indices[i*subgroup_size : (i+1)*subgroup_size] for i in range(k)]
    
    xbars = groups.mean(axis=1)
    ranges = np.ptp(groups, axis=1)

    grouping_column_subgroups = [
        df.loc[indices_subgroups[i], grouping_column].tolist() 
        for i in range(k)
    ]
    
    Xbar_bar = xbars.mean()
    R_bar = ranges.mean()
    constants = {
        'A2': {2:1.88,3:1.023,4:0.729,5:0.577,6:0.483,7:0.419,8:0.373,9:0.337,10:0.308},
        'D3': {2:0,3:0,4:0,5:0,6:0,7:0.076,8:0.136,9:0.184,10:0.223},
        'D4': {2:3.267,3:2.574,4:2.282,5:2.114,6:2.004,7:1.924,8:1.864,9:1.816,10:1.777}
    }
    
    try:
        A2 = constants['A2'][subgroup_size]
        D3 = constants['D3'][subgroup_size]
        D4 = constants['D4'][subgroup_size]
    except KeyError:
        raise ValueError(f"Subgroup size {subgroup_size} not supported. Valid sizes: 2-10")

    UCL_x = Xbar_bar + A2 * R_bar
    LCL_x = Xbar_bar - A2 * R_bar
    UCL_R = D4 * R_bar
    LCL_R = D3 * R_bar

    fig = make_subplots(
        rows=2, cols=1,
        #xshared=True,
        vertical_spacing=0.15,
        subplot_titles=(
            f'X-bar Chart (Subgroup Size: {subgroup_size})',
            f'R Chart (Subgroup Size: {subgroup_size})'
        )
    )

    hover_template = (
        "<b>Subgroup %{x}</b><br>"
        "Value: %{y:.2f}<br>"
        "%{grouping_column}: %{customdata}<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(k),
            y=xbars,
            mode='lines+markers',
            name='X-bar',
            customdata=[f"{', '.join(map(str, ms))}" for ms in grouping_column_subgroups],
            hovertemplate=hover_template,
            marker=dict(color='#1f77b4'),
            line=dict(width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(k),
            y=np.atleast_1d(ranges),
            mode='lines+markers',
            name='R',
            customdata=[f"{', '.join(map(str, ms))}" for ms in grouping_column_subgroups],
            hovertemplate=hover_template,
            marker=dict(color='#ff7f0e'),
            line=dict(width=1)
        ),
        row=2, col=1
    )

    for row, limits in enumerate([(UCL_x, LCL_x, Xbar_bar), (UCL_R, LCL_R, R_bar)], 1):
        fig.add_hline(
            y=limits[0], line=dict(color='red', dash='dash'),
            row=row, col=1,
            annotation_text=f'UCL: {limits[0]:.2f}',
            annotation_position="top right"
        )
        fig.add_hline(
            y=limits[1], line=dict(color='red', dash='dash'),
            row=row, col=1,
            annotation_text=f'LCL: {limits[1]:.2f}',
            annotation_position="bottom right"
        )
        fig.add_hline(
            y=limits[2], line=dict(color='black'),
            row=row, col=1,
            annotation_text=f'CL: {limits[2]:.2f}',
            annotation_position="top left"
        )

    fig.update_layout(
        height=800,
        title_text="Interactive X-bar and R Control Charts",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        margin=dict(t=100),
        xaxis2=dict(title='Subgroup Index')
    )

    fig.update_yaxes(title_text="Subgroup Mean", row=1, col=1)
    fig.update_yaxes(title_text="Range", row=2, col=1)

    fig.update_xaxes(matches='x', showticklabels=True)
    fig.update_layout(
        xaxis=dict(showticklabels=False), 
        xaxis2=dict(title='Subgroup Index')
    )

    return fig



if __name__ == '__main__':
    pass

