import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
from functools import lru_cache
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, name='Temporal Analysis', icon='clock')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    df['year'] = df['Date and Time'].dt.year
    df['month'] = df['Date and Time'].dt.month
    df['day'] = df['Date and Time'].dt.day
    df['hour'] = df['Date and Time'].dt.hour
    df['weekday'] = df['Date and Time'].dt.dayofweek
    df['quarter'] = df['Date and Time'].dt.quarter
    df['day_of_year'] = df['Date and Time'].dt.dayofyear
    df['week_of_year'] = df['Date and Time'].dt.isocalendar().week
    return df

df = load_data()

# --- Create Enhanced Temporal Figures ---

# 1. Long-term Earthquake Activity Timeline (SPLIT INTO 3 FIGURES)
df_yearly = df.groupby('year').agg({
    'MAG': ['count', 'mean', 'max', 'std'],
    'DEPTH': 'mean'
}).round(2)
df_yearly.columns = ['_'.join(col).strip() for col in df_yearly.columns]
df_yearly = df_yearly.reset_index()

fig_timeline_count = go.Figure(go.Scatter(
    x=df_yearly['year'],
    y=df_yearly['MAG_count'],
    mode='lines+markers',
    line=dict(color='#58a6ff', width=3),
    marker=dict(size=8),
    name='Count',
    hovertemplate='<b>Year:</b> %{x}<br><b>Earthquakes:</b> %{y}<extra></extra>'
))
fig_timeline_count.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Yearly Earthquake Count', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Year',
    yaxis_title='Count',
    showlegend=False
)

fig_timeline_avg = go.Figure(go.Scatter(
    x=df_yearly['year'],
    y=df_yearly['MAG_mean'],
    mode='lines+markers',
    line=dict(color='#2ea043', width=3),
    marker=dict(size=8),
    name='Avg Magnitude',
    hovertemplate='<b>Year:</b> %{x}<br><b>Avg Magnitude:</b> %{y:.2f}<extra></extra>'
))
fig_timeline_avg.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Average Magnitude per Year', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Year',
    yaxis_title='Average Magnitude',
    showlegend=False
)

fig_timeline_max = go.Figure(go.Scatter(
    x=df_yearly['year'],
    y=df_yearly['MAG_max'],
    mode='lines+markers',
    line=dict(color='#f85149', width=3),
    marker=dict(size=8),
    name='Max Magnitude',
    hovertemplate='<b>Year:</b> %{x}<br><b>Max Magnitude:</b> %{y:.1f}<extra></extra>'
))
fig_timeline_max.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Maximum Magnitude per Year', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Year',
    yaxis_title='Maximum Magnitude',
    showlegend=False
)

# 2. Seasonal and Cyclical Patterns (SPLIT INTO 4 FIGURES)
monthly_stats = df.groupby('month').agg({'MAG': ['count', 'mean']}).round(2)
monthly_stats.columns = ['count', 'avg_mag']
monthly_stats = monthly_stats.reset_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig_seasonal_month = go.Figure(go.Bar(
    x=[month_names[i-1] for i in monthly_stats['month']],
    y=monthly_stats['count'],
    marker_color='#58a6ff',
    name='Monthly Count',
    hovertemplate='<b>Month:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
))
fig_seasonal_month.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Monthly Pattern', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Month',
    yaxis_title='Count',
    showlegend=False
)

quarterly_stats = df.groupby('quarter').agg({'MAG': ['count', 'mean']}).round(2)
quarterly_stats.columns = ['count', 'avg_mag']
quarterly_stats = quarterly_stats.reset_index()
fig_seasonal_quarter = go.Figure(go.Bar(
    x=['Q1', 'Q2', 'Q3', 'Q4'],
    y=quarterly_stats['count'],
    marker_color='#2ea043',
    name='Quarterly Count',
    hovertemplate='<b>Quarter:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
))
fig_seasonal_quarter.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Quarterly Pattern', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Quarter',
    yaxis_title='Count',
    showlegend=False
)

weekly_stats = df.groupby('weekday').agg({'MAG': ['count', 'mean']}).round(2)
weekly_stats.columns = ['count', 'avg_mag']
weekly_stats = weekly_stats.reset_index()
weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
fig_seasonal_week = go.Figure(go.Bar(
    x=[weekday_names[i] for i in weekly_stats['weekday']],
    y=weekly_stats['count'],
    marker_color='#d29922',
    name='Weekly Count',
    hovertemplate='<b>Day:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
))
fig_seasonal_week.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Weekly Pattern', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Day',
    yaxis_title='Count',
    showlegend=False
)

hourly_stats = df.groupby('hour').agg({'MAG': ['count', 'mean']}).round(2)
hourly_stats.columns = ['count', 'avg_mag']
hourly_stats = hourly_stats.reset_index()
fig_seasonal_hour = go.Figure(go.Bar(
    x=hourly_stats['hour'],
    y=hourly_stats['count'],
    marker_color='#f85149',
    name='Hourly Count',
    hovertemplate='<b>Hour:</b> %{x}:00<br><b>Count:</b> %{y}<extra></extra>'
))
fig_seasonal_hour.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Hourly Pattern', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Hour',
    yaxis_title='Count',
    showlegend=False
)

# 3. Earthquake Sequence Analysis (SPLIT INTO 2 FIGURES)
df_sorted = df.sort_values('Date and Time')
df_sorted['time_diff'] = df_sorted['Date and Time'].diff().dt.total_seconds() / 3600  # hours
df_sorted['cumulative_count'] = range(1, len(df_sorted) + 1)

fig_sequence_cum = go.Figure(go.Scatter(
    x=df_sorted['Date and Time'],
    y=df_sorted['cumulative_count'],
    mode='lines',
    line=dict(color='#58a6ff', width=2),
    name='Cumulative Count',
    hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Count:</b> %{y}<extra></extra>'
))
fig_sequence_cum.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Cumulative Earthquake Count Over Time', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Date',
    yaxis_title='Cumulative Count',
    showlegend=False
)

fig_sequence_interval = go.Figure(go.Histogram(
    x=np.log10(df_sorted['time_diff'].dropna()),
    nbinsx=50,
    marker_color='#2ea043',
    opacity=0.8,
    name='Log Time Intervals',
    hovertemplate='<b>Log10(Hours):</b> %{x:.2f}<br><b>Count:</b> %{y}<extra></extra>'
))
fig_sequence_interval.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Time Intervals Between Earthquakes', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 16, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=60, l=60, r=40),
    xaxis_title='Log10(Hours between events)',
    yaxis_title='Frequency',
    showlegend=False
)

# 4. Magnitude Evolution Analysis
# Rolling statistics
window_size = 30
df_sorted['mag_rolling_mean'] = df_sorted['MAG'].rolling(window=window_size, center=True).mean()
df_sorted['mag_rolling_std'] = df_sorted['MAG'].rolling(window=window_size, center=True).std()
df_sorted['mag_rolling_max'] = df_sorted['MAG'].rolling(window=window_size, center=True).max()
df_sorted['mag_rolling_min'] = df_sorted['MAG'].rolling(window=window_size, center=True).min()

fig_evolution = go.Figure()

# Add rolling mean
fig_evolution.add_trace(go.Scatter(
    x=df_sorted['Date and Time'],
    y=df_sorted['mag_rolling_mean'],
    mode='lines',
    line=dict(color='#58a6ff', width=3),
    name=f'{window_size}-Event Rolling Mean',
    hovertemplate='<b>Date:</b> %{x}<br>' +
                  '<b>Rolling Mean:</b> %{y:.2f}<extra></extra>'
))

# Add confidence bands
fig_evolution.add_trace(go.Scatter(
    x=df_sorted['Date and Time'],
    y=df_sorted['mag_rolling_mean'] + df_sorted['mag_rolling_std'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_evolution.add_trace(go.Scatter(
    x=df_sorted['Date and Time'],
    y=df_sorted['mag_rolling_mean'] - df_sorted['mag_rolling_std'],
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(88, 166, 255, 0.2)',
    name='±1 Std Dev',
    hoverinfo='skip'
))

# Add individual earthquakes
fig_evolution.add_trace(go.Scatter(
    x=df_sorted['Date and Time'],
    y=df_sorted['MAG'],
    mode='markers',
    marker=dict(
        size=4,
        color='rgba(249, 81, 73, 0.6)',
        line=dict(width=0)
    ),
    name='Individual Earthquakes',
    hovertemplate='<b>Date:</b> %{x}<br>' +
                  '<b>Magnitude:</b> %{y:.1f}<extra></extra>'
))

fig_evolution.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=700,
    title={'text': f'Magnitude Evolution Analysis ({window_size}-Event Rolling Window)', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=100, b=100, l=80, r=80),  # increased bottom margin
    yaxis_title='Magnitude',
    xaxis_title='Date',
    xaxis=dict(
        title='Date',
        showgrid=True,
        showticklabels=True,
        showline=True,
        linecolor='#f0f6fc',
        tickangle=45
    )
)

# 5. Frequency-Magnitude Relationship (Gutenberg-Richter)
mag_bins = np.arange(df['MAG'].min(), df['MAG'].max() + 0.1, 0.1)
mag_counts = []
for i in range(len(mag_bins) - 1):
    count = len(df[df['MAG'] >= mag_bins[i]])
    mag_counts.append(count)

# Fit linear regression in log space
log_counts = np.log10(mag_counts)
valid_indices = np.isfinite(log_counts) & (np.array(mag_counts) > 0)
if np.sum(valid_indices) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(mag_bins[:-1][valid_indices], log_counts[valid_indices])
    fitted_line = slope * mag_bins[:-1] + intercept
else:
    fitted_line = np.zeros_like(mag_bins[:-1])
    slope, r_value = 0, 0

fig_gutenberg = go.Figure()

# Original data
fig_gutenberg.add_trace(go.Scatter(
    x=mag_bins[:-1],
    y=mag_counts,
    mode='markers',
    marker=dict(size=8, color='#58a6ff'),
    name='Observed',
    hovertemplate='<b>Magnitude ≥:</b> %{x:.1f}<br>' +
                  '<b>Count:</b> %{y}<extra></extra>'
))

# Fitted line
if slope != 0:
    fig_gutenberg.add_trace(go.Scatter(
        x=mag_bins[:-1],
        y=10**fitted_line,
        mode='lines',
        line=dict(color='#f85149', width=3, dash='dash'),
        name=f'G-R Fit (b={-slope:.2f}, R²={r_value**2:.3f})',
        hovertemplate='<b>Magnitude ≥:</b> %{x:.1f}<br>' +
                      '<b>Fitted Count:</b> %{y:.0f}<extra></extra>'
    ))

fig_gutenberg.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=700,
    title={'text': 'Gutenberg-Richter Frequency-Magnitude Relationship', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=100, b=100, l=80, r=80),
    yaxis_title='Cumulative Number of Earthquakes',
    xaxis_title='Magnitude',
    yaxis_type='log'
)

# 6. Recurrence Analysis
def calculate_recurrence_intervals(magnitudes, threshold):
    """Calculate recurrence intervals for earthquakes above threshold"""
    above_threshold = df_sorted[df_sorted['MAG'] >= threshold]['Date and Time']
    if len(above_threshold) < 2:
        return []
    intervals = above_threshold.diff().dt.total_seconds() / (365.25 * 24 * 3600)  # years
    return intervals.dropna().values

thresholds = [4.0, 4.5, 5.0, 5.5, 6.0]
fig_recurrence = go.Figure()

colors = ['#58a6ff', '#2ea043', '#d29922', '#f85149', '#8b5cf6']
for i, threshold in enumerate(thresholds):
    intervals = calculate_recurrence_intervals(df_sorted['MAG'], threshold)
    if len(intervals) > 0:
        fig_recurrence.add_trace(go.Histogram(
            x=intervals,
            nbinsx=20,
            name=f'M ≥ {threshold}',
            opacity=0.7,
            marker_color=colors[i],
            hovertemplate=f'<b>M ≥ {threshold}</b><br>' +
                          '<b>Recurrence Interval:</b> %{x:.2f} years<br>' +
                          '<b>Count:</b> %{y}<extra></extra>'
        ))

fig_recurrence.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=700,
    title={'text': 'Earthquake Recurrence Intervals', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=100, b=100, l=80, r=80),
    xaxis_title='Recurrence Interval (years)',
    yaxis_title='Frequency',
    barmode='overlay'
)

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-clock me-3 text-accent"),
            "Temporal Analysis"
        ], className="mb-0 fw-bold"),
        html.P("Comprehensive temporal analysis of earthquake patterns, trends, and cyclical behaviors", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Long-term Trends Section
    html.Div([
        html.H2([
            html.I (className="fas fa-chart-line me-2 text-accent"),
            "Long-term Trends"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-calendar-alt me-2"),
                            "Earthquake Activity Timeline"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Long-term trends in earthquake frequency, average magnitude, and maximum magnitude per year", 
                               className="text-muted mb-3"),
                        dcc.Graph(figure=fig_timeline_count, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_timeline_avg, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_timeline_max, config={'displayModeBar': True, 'displaylogo': False})
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-arrow-trend-up me-2"),
                            "Magnitude Evolution"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Evolution of earthquake magnitudes over time with rolling statistics and confidence intervals", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_evolution,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Cyclical Patterns Section
    html.Div([
        html.H2([
            html.I (className="fas fa-sync-alt me-2 text-accent"),
            "Cyclical and Seasonal Patterns"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-calendar me-2"),
                            "Seasonal Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Monthly, quarterly, weekly, and hourly patterns in earthquake occurrence", 
                               className="text-muted mb-3"),
                        dcc.Graph(figure=fig_seasonal_month, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_seasonal_quarter, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_seasonal_week, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_seasonal_hour, config={'displayModeBar': True, 'displaylogo': False})
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Sequence Analysis Section
    html.Div([
        html.H2([
            html.I (className="fas fa-list-ol me-2 text-accent"),
            "Sequence Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-step-forward me-2"),
                            "Earthquake Sequences"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Cumulative earthquake count over time and distribution of time intervals between events", 
                               className="text-muted mb-3"),
                        dcc.Graph(figure=fig_sequence_cum, config={'displayModeBar': True, 'displaylogo': False}),
                        dcc.Graph(figure=fig_sequence_interval, config={'displayModeBar': True, 'displaylogo': False})
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Statistical Analysis Section
    html.Div([
        html.H2([
            html.I (className="fas fa-chart-bar me-2 text-accent"),
            "Statistical Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-wave-square me-2"),
                            "Gutenberg-Richter Relationship"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Frequency-magnitude relationship following the Gutenberg-Richter law with fitted parameters", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_gutenberg,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I (className="fas fa-redo me-2"),
                            "Recurrence Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Distribution of recurrence intervals for different magnitude thresholds", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_recurrence,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6)
        ])
    ])
], className="page-content")
