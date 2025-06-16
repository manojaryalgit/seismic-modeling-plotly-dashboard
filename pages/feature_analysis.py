import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
from functools import lru_cache

dash.register_page(__name__, name='Feature Analysis', icon='chart-bar')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    df['month_name'] = df['Date and Time'].dt.month_name()
    return df

df = load_data()

# --- Create Enhanced Figures ---
fig_b_value = px.line(
    df, 
    x='Date and Time', 
    y='b_value_1', 
    title='Gutenberg-Richter b-value Over Time', 
    labels={'b_value_1': 'b-value', 'Date and Time': 'Date'}, 
    template="plotly_dark"
)
fig_b_value.update_traces(line=dict(color='#58a6ff', width=3))
fig_b_value.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': '#f0f6fc'}
    },
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    plot_bgcolor='rgba(28, 33, 40, 0.95)',
    margin=dict(t=60, b=40, l=60, r=40),
    height=360,
    xaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    ),
    yaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    )
)

fig_energy = px.line(
    df, 
    x='Date and Time', 
    y='sqrt_Energy_Release', 
    title='Cumulative Seismic Energy Release (sqrt)', 
    labels={'sqrt_Energy_Release': 'Sqrt of Energy', 'Date and Time': 'Date'}, 
    template="plotly_dark"
)
fig_energy.update_traces(line=dict(color='#2ea043', width=3))
fig_energy.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': '#f0f6fc'}
    },
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    plot_bgcolor='rgba(28, 33, 40, 0.95)',
    margin=dict(t=60, b=40, l=60, r=40),
    height=360,
    xaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    ),
    yaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    )
)

fig_mag_dist = px.histogram(
    df, 
    x="MAG", 
    nbins=50, 
    title="Distribution of Earthquake Magnitudes", 
    template="plotly_dark",
    labels={'MAG': 'Magnitude', 'count': 'Frequency'}
)
fig_mag_dist.update_traces(
    marker=dict(
        color='#d29922', 
        opacity=0.8,
        line=dict(width=1, color='#f0f6fc')
    )
)
fig_mag_dist.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': '#f0f6fc'}
    },
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    plot_bgcolor='rgba(28, 33, 40, 0.95)',
    margin=dict(t=60, b=40, l=60, r=40),
    height=310,
    xaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    ),
    yaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    )
)

monthly_counts = df['month_name'].value_counts().reset_index()
monthly_counts.columns = ['month_name', 'count']
month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
monthly_counts['month_name'] = pd.Categorical(monthly_counts['month_name'], categories=month_order, ordered=True)
monthly_counts = monthly_counts.sort_values('month_name')

fig_monthly_dist = px.bar(
    monthly_counts, 
    x='month_name', 
    y='count', 
    title="Earthquakes by Month", 
    template="plotly_dark",
    labels={'month_name': 'Month', 'count': 'Number of Earthquakes'},
    color='count',
    color_continuous_scale=[[0, '#2ea043'], [0.5, '#d29922'], [1, '#f85149']]
)
fig_monthly_dist.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'color': '#f0f6fc'}
    },
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    plot_bgcolor='rgba(28, 33, 40, 0.95)',
    margin=dict(t=60, b=40, l=60, r=40),
    height=310,
    xaxis_tickangle=-45,
    xaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    ),
    yaxis=dict(
        gridcolor='rgba(48, 54, 61, 0.5)',
        color='#f0f6fc'
    ),
    coloraxis_colorbar=dict(
        title="Count",
        title_font_color='#f0f6fc',
        tickfont_color='#f0f6fc'
    )
)
fig_monthly_dist.update_traces(
    marker_line_color='#f0f6fc',
    marker_line_width=1
)


# --- Enhanced Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-bar me-3 text-accent"),
            "Feature Analysis"
        ], className="mb-0 fw-bold"),
        html.P("Detailed analysis of seismic features and patterns for earthquake prediction modeling", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Time Series Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-clock me-2 text-accent"),
            "Time Series Analysis"
        ], className="section-title mb-4"),
        
        # B-value analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-wave-square me-2"),
                            "Gutenberg-Richter b-value"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("The b-value represents the relative number of small to large earthquakes and is crucial for seismic hazard assessment.", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_b_value,
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '400px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5"),
        
        # Energy release analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-bolt me-2"),
                            "Seismic Energy Release"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Cumulative seismic energy release shows the overall earthquake activity and helps identify periods of increased seismic activity.", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_energy,
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '400px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Distribution Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-chart-pie me-2 text-accent"),
            "Distribution Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            # Magnitude distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-area me-2"),
                            "Magnitude Distribution"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Distribution of earthquake magnitudes follows the Gutenberg-Richter law with more frequent smaller earthquakes.", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_mag_dist,
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '350px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6),
            
            # Monthly distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-calendar-alt me-2"),
                            "Seasonal Patterns"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Monthly earthquake frequency analysis to identify potential seasonal patterns in seismic activity.", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_monthly_dist,
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '350px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6)
        ])
    ])
], className="page-content")
