import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np



def plot_xbar_s_chart(
        df, 
        value_column: str=None,
        grouping_column: str=None, 
        subgroup_size=5):
    """
    Interactive X̄ & S chart with synchronized zoom/hover using Plotly.
    - df: pandas DataFrame
    - value_column: name of the measurement value_column
    - subgroup_size: number of observations per subgroup
    """
    # Data preparation
    data_series = df[value_column].dropna()
    data = data_series.values
    indices = data_series.index
    
    # Split into subgroups
    k = len(data) // subgroup_size
    if k == 0:
        raise ValueError("Subgroup size is larger than the number of non-NA data points.")
    
    groups = data[:k*subgroup_size].reshape((k, subgroup_size))
    indices_subgroups = [indices[i*subgroup_size : (i+1)*subgroup_size] for i in range(k)]
    
    # Calculate subgroup statistics
    xbars = groups.mean(axis=1)
    s_values = np.std(groups, axis=1, ddof=1)  # Sample standard deviation
    
    # Get material_salida values
    material_salida_subgroups = [
        df.loc[indices_subgroups[i], grouping_column].tolist() 
        for i in range(k)
    ]
    
    # Control limit calculations
    Xbar_bar = xbars.mean()
    S_bar = s_values.mean()
    
    # Constants for S chart (ASTM tables)
    constants = {
        'A3': {2:2.659, 3:1.954, 4:1.628, 5:1.427, 6:1.287, 7:1.182, 8:1.099, 9:1.032, 10:0.975},
        'B3': {2:0, 3:0, 4:0, 5:0, 6:0.030, 7:0.118, 8:0.185, 9:0.239, 10:0.284},
        'B4': {2:3.267, 3:2.568, 4:2.266, 5:2.089, 6:1.970, 7:1.882, 8:1.815, 9:1.761, 10:1.716}
    }
    
    try:
        A3 = constants['A3'][subgroup_size]
        B3 = constants['B3'][subgroup_size]
        B4 = constants['B4'][subgroup_size]
    except KeyError:
        raise ValueError(f"Subgroup size {subgroup_size} not supported. Valid sizes: 2-10")

    # Control limits
    UCL_x = Xbar_bar + A3 * S_bar
    LCL_x = Xbar_bar - A3 * S_bar
    UCL_S = B4 * S_bar
    LCL_S = B3 * S_bar

    # Create figure with synchronized axes
    fig = make_subplots(
        rows=2, cols=1,
        #shared_x=True,
        vertical_spacing=0.15,
        subplot_titles=(
            f'X-bar Chart (Subgroup Size: {subgroup_size})',
            f'S Chart (Subgroup Size: {subgroup_size})'
        )
    )

    # Custom hover template
    hover_template = (
        "<b>Subgroup %{x}</b><br>"
        "Value: %{y:.2f}<br>"
        "{grouping_column}: %{customdata}<extra></extra>"
    )

    # X-bar Chart
    fig.add_trace(
        go.Scatter(
            x=np.arange(k),
            y=xbars,
            mode='lines+markers',
            name='X-bar',
            customdata=[", ".join(map(str, ms)) for ms in material_salida_subgroups],
            hovertemplate=hover_template,
            marker=dict(color='#1f77b4'),
            line=dict(width=1)
        ),
        row=1, col=1
    )

    # S Chart
    fig.add_trace(
        go.Scatter(
            x=np.arange(k),
            y=s_values,
            mode='lines+markers',
            name='S',
            customdata=[", ".join(map(str, ms)) for ms in material_salida_subgroups],
            hovertemplate=hover_template,
            marker=dict(color='#2ca02c'),
            line=dict(width=1)
        ),
        row=2, col=1
    )

    # Add control limits as shapes
    for row, (UCL, LCL, CL, chart_type) in enumerate(zip(
        [UCL_x, UCL_S], [LCL_x, LCL_S], [Xbar_bar, S_bar], ['X-bar', 'S']
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
        title_text="Interactive X-bar and S Control Charts",
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        margin=dict(t=100),
        xaxis2=dict(title='Subgroup Index')
    )
    
    # Axis labels
    fig.update_yaxes(title_text="Subgroup Mean", row=1, col=1)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)

    # Synchronized zoom configuration
    fig.update_xaxes(matches='x', showticklabels=True)
    fig.update_layout(
        xaxis=dict(showticklabels=False),  # Hide x-axis labels on top plot
        xaxis2=dict(title='Subgroup Index')
    )

    return fig


if __name__ == '__main__':
    pass