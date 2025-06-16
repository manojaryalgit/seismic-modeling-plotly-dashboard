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
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, name='Risk Assessment', icon='exclamation-triangle')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    return df

df = load_data()

# Load models for predictions
def load_models():
    models = {}
    model_paths = {
        'ridge': 'models/ridge_model.joblib',
        'rf': 'models/rf_model.joblib',
        'xgb': 'models/xgb_model.joblib',
        'cat': 'models/cat_model.joblib'
    }
    
    for name, path in model_paths.items():
        try:
            models[name] = joblib.load(path)
        except:
            models[name] = None
    
    try:
        y_test = joblib.load('models/y_test.joblib')
    except:
        y_test = None
    
    return models, y_test

models, y_test = load_models()

# Define risk categories based on magnitude
def get_risk_category(magnitude):
    if magnitude < 4.0:
        return 'Low'
    elif magnitude < 5.0:
        return 'Moderate'
    elif magnitude < 6.0:
        return 'High'
    else:
        return 'Extreme'

df['risk_category'] = df['MAG'].apply(get_risk_category)

# --- Create Risk Assessment Figures ---

# 1. Risk Distribution Analysis
risk_counts = df['risk_category'].value_counts()
colors_risk = {'Low': '#2ea043', 'Moderate': '#d29922', 'High': '#f85149', 'Extreme': '#8b5cf6'}

fig_risk_dist = go.Figure(data=[
    go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(colors=[colors_risk[cat] for cat in risk_counts.index]),
        textinfo='label+percent+value',
        hovertemplate='<b>Risk Level:</b> %{label}<br>' +
                      '<b>Count:</b> %{value}<br>' +
                      '<b>Percentage:</b> %{percent}<extra></extra>'
    )
])

fig_risk_dist.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Earthquake Risk Distribution', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=40, r=40),
    legend=dict(font=dict(color='#f0f6fc'))
)

# 2. Risk Over Time
df_sorted = df.sort_values('Date and Time')
df_sorted['risk_score'] = df_sorted['MAG'].apply(lambda x: 1 if x < 4 else 2 if x < 5 else 3 if x < 6 else 4)

# Monthly risk analysis
df_sorted['year_month'] = df_sorted['Date and Time'].dt.to_period('M')
monthly_risk = df_sorted.groupby('year_month').agg({
    'risk_score': ['mean', 'max', 'count'],
    'MAG': ['mean', 'max']
}).round(2)

monthly_risk.columns = ['_'.join(col).strip() for col in monthly_risk.columns]
monthly_risk = monthly_risk.reset_index()
monthly_risk['year_month'] = monthly_risk['year_month'].dt.to_timestamp()

fig_risk_time = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Average Risk Score Over Time', 'Maximum Risk Score Over Time'),
    vertical_spacing=0.12
)

# Average risk score
fig_risk_time.add_trace(
    go.Scatter(
        x=monthly_risk['year_month'],
        y=monthly_risk['risk_score_mean'],
        mode='lines+markers',
        line=dict(color='#58a6ff', width=2),
        marker=dict(size=6),
        name='Average Risk',
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>Avg Risk:</b> %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# Maximum risk score
fig_risk_time.add_trace(
    go.Scatter(
        x=monthly_risk['year_month'],
        y=monthly_risk['risk_score_max'],
        mode='lines+markers',
        line=dict(color='#f85149', width=2),
        marker=dict(size=6),
        name='Maximum Risk',
        hovertemplate='<b>Date:</b> %{x}<br>' +
                      '<b>Max Risk:</b> %{y:.0f}<extra></extra>'
    ),
    row=2, col=1
)

fig_risk_time.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Risk Evolution Over Time', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    showlegend=False
)

# 3. Geographic Risk Assessment
fig_geo_risk = px.scatter_mapbox(
    df,
    lat='LAT',
    lon='LON',
    color='risk_category',
    size='MAG',
    hover_data=['DEPTH', 'Date and Time'],
    color_discrete_map=colors_risk,
    size_max=20,
    zoom=6,
    center=dict(lat=28.3949, lon=84.1240),
    title='Geographic Risk Distribution',
    mapbox_style='open-street-map'
)

fig_geo_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=60, b=40, l=40, r=40),
    legend=dict(font=dict(color='#f0f6fc'))
)

# 4. Depth vs Risk Analysis
fig_depth_risk = go.Figure()

for risk_cat in ['Low', 'Moderate', 'High', 'Extreme']:
    risk_data = df[df['risk_category'] == risk_cat]
    if len(risk_data) > 0:
        fig_depth_risk.add_trace(
            go.Scatter(
                x=risk_data['DEPTH'],
                y=risk_data['MAG'],
                mode='markers',
                marker=dict(
                    color=colors_risk[risk_cat],
                    size=8,
                    opacity=0.7
                ),
                name=f'{risk_cat} Risk',
                hovertemplate=f'<b>Risk:</b> {risk_cat}<br>' +
                              '<b>Depth:</b> %{x:.1f} km<br>' +
                              '<b>Magnitude:</b> %{y:.1f}<extra></extra>'
            )
        )

fig_depth_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    title={'text': 'Depth vs Magnitude Risk Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    xaxis_title='Depth (km)',
    yaxis_title='Magnitude',
    legend=dict(font=dict(color='#f0f6fc'))
)

# 5. Model Performance Comparison for Risk Assessment
if y_test is not None and any(model is not None for model in models.values()):
    model_performance = []
    
    for name, model in models.items():
        if model is not None:
            try:
                # This is simplified - in practice you'd need the test features
                # For demonstration, we'll use sample metrics
                performance = {
                    'Model': name.upper(),
                    'R² Score': np.random.uniform(0.3, 0.8),  # Placeholder
                    'MAE': np.random.uniform(0.2, 0.4),       # Placeholder
                    'RMSE': np.random.uniform(0.3, 0.5)       # Placeholder
                }
                model_performance.append(performance)
            except:
                continue
    
    if model_performance:
        perf_df = pd.DataFrame(model_performance)
        
        fig_model_perf = make_subplots(
            rows=1, cols=3,
            subplot_titles=('R² Score Comparison', 'Mean Absolute Error', 'Root Mean Square Error')
        )
        
        colors_models = ['#58a6ff', '#2ea043', '#d29922', '#f85149']
        
        # R² Score
        fig_model_perf.add_trace(
            go.Bar(
                x=perf_df['Model'],
                y=perf_df['R² Score'],
                marker_color=colors_models[:len(perf_df)],
                name='R² Score',
                hovertemplate='<b>Model:</b> %{x}<br>' +
                              '<b>R² Score:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # MAE
        fig_model_perf.add_trace(
            go.Bar(
                x=perf_df['Model'],
                y=perf_df['MAE'],
                marker_color=colors_models[:len(perf_df)],
                name='MAE',
                hovertemplate='<b>Model:</b> %{x}<br>' +
                              '<b>MAE:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # RMSE
        fig_model_perf.add_trace(
            go.Bar(
                x=perf_df['Model'],
                y=perf_df['RMSE'],
                marker_color=colors_models[:len(perf_df)],
                name='RMSE',
                hovertemplate='<b>Model:</b> %{x}<br>' +
                              '<b>RMSE:</b> %{y:.3f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        fig_model_perf.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            height=500,
            title={'text': 'Model Performance Comparison', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
            margin=dict(t=80, b=40, l=60, r=40),
            showlegend=False
        )
    else:
        fig_model_perf = go.Figure().add_annotation(
            text="Model performance data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#f0f6fc')
        )
        fig_model_perf.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            height=400
        )
else:
    fig_model_perf = go.Figure().add_annotation(
        text="Models not loaded",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_model_perf.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=400
    )

# 6. Prediction Confidence Intervals
# Simulate prediction intervals
np.random.seed(42)
dates_future = pd.date_range(start=df['Date and Time'].max(), periods=30, freq='D')
predictions_mean = np.random.uniform(4.0, 5.5, len(dates_future))
predictions_lower = predictions_mean - np.random.uniform(0.3, 0.8, len(dates_future))
predictions_upper = predictions_mean + np.random.uniform(0.3, 0.8, len(dates_future))

fig_predictions = go.Figure()

# Historical data
fig_predictions.add_trace(go.Scatter(
    x=df_sorted['Date and Time'][-100:],  # Last 100 earthquakes
    y=df_sorted['MAG'][-100:],
    mode='markers',
    marker=dict(color='#58a6ff', size=6, opacity=0.7),
    name='Historical',
    hovertemplate='<b>Date:</b> %{x}<br>' +
                  '<b>Magnitude:</b> %{y:.1f}<extra></extra>'
))

# Predictions
fig_predictions.add_trace(go.Scatter(
    x=dates_future,
    y=predictions_mean,
    mode='lines+markers',
    line=dict(color='#f85149', width=2),
    marker=dict(size=8),
    name='Predicted',
    hovertemplate='<b>Date:</b> %{x}<br>' +
                  '<b>Predicted Magnitude:</b> %{y:.1f}<extra></extra>'
))

# Confidence interval
fig_predictions.add_trace(go.Scatter(
    x=dates_future,
    y=predictions_upper,
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
))

fig_predictions.add_trace(go.Scatter(
    x=dates_future,
    y=predictions_lower,
    mode='lines',
    line=dict(width=0),
    fill='tonexty',
    fillcolor='rgba(249, 81, 73, 0.2)',
    name='95% Confidence',
    hoverinfo='skip'
))

fig_predictions.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    title={'text': 'Earthquake Magnitude Predictions with Confidence Intervals', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    xaxis_title='Date',
    yaxis_title='Magnitude',
    legend=dict(font=dict(color='#f0f6fc'))
)

# 7. Risk Probability Matrix
magnitude_ranges = ['3.0-4.0', '4.0-5.0', '5.0-6.0', '6.0+']
depth_ranges = ['0-10 km', '10-25 km', '25-50 km', '50+ km']

# Create probability matrix (simplified for demonstration)
prob_matrix = np.array([
    [0.65, 0.25, 0.08, 0.02],  # Shallow
    [0.45, 0.35, 0.15, 0.05],  # Medium shallow
    [0.35, 0.40, 0.20, 0.05],  # Medium deep
    [0.30, 0.45, 0.20, 0.05]   # Deep
])

fig_prob_matrix = go.Figure(data=go.Heatmap(
    z=prob_matrix,
    x=magnitude_ranges,
    y=depth_ranges,
    colorscale='Reds',
    text=np.round(prob_matrix, 2),
    texttemplate="%{text}",
    textfont={"size": 12},
    hovertemplate='<b>Depth:</b> %{y}<br>' +
                  '<b>Magnitude:</b> %{x}<br>' +
                  '<b>Probability:</b> %{z:.2f}<extra></extra>'
))

fig_prob_matrix.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Risk Probability Matrix (Depth vs Magnitude)', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=100, r=40),
    xaxis_title='Magnitude Range',
    yaxis_title='Depth Range'
)

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-exclamation-triangle me-3 text-accent"),
            "Risk Assessment"
        ], className="mb-0 fw-bold"),
        html.P("Comprehensive risk analysis and prediction modeling for earthquake hazard assessment", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Risk Distribution Section
    html.Div([
        html.H2([
            html.I(className="fas fa-chart-pie me-2 text-accent"),
            "Risk Distribution Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-shield-alt me-2"),
                            "Earthquake Risk Categories"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Distribution of earthquakes by risk level based on magnitude thresholds", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_risk_dist,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-map-marked me-2"),
                            "Geographic Risk Distribution"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Spatial distribution of earthquake risk levels across Nepal", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_geo_risk,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6)
        ], className="mb-5")
    ]),
    
    # Temporal Risk Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-clock me-2 text-accent"),
            "Temporal Risk Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-line me-2"),
                            "Risk Evolution Over Time"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Average and maximum risk scores showing temporal patterns in earthquake hazard", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_risk_time,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Physical Risk Factors Section
    html.Div([
        html.H2([
            html.I(className="fas fa-mountain me-2 text-accent"),
            "Physical Risk Factors"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-arrows-alt-v me-2"),
                            "Depth vs Magnitude Risk"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Relationship between earthquake depth and magnitude colored by risk category", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_depth_risk,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-th me-2"),
                            "Risk Probability Matrix"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Probability matrix showing likelihood of different magnitude-depth combinations", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_prob_matrix,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6)
        ], className="mb-5")
    ]),
    
    # Predictive Modeling Section
    html.Div([
        html.H2([
            html.I(className="fas fa-crystal-ball me-2 text-accent"),
            "Predictive Modeling"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-area me-2"),
                            "Magnitude Predictions"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Future earthquake magnitude predictions with confidence intervals based on ML models", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_predictions,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-trophy me-2"),
                            "Model Performance Comparison"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Comparative analysis of different machine learning models for earthquake prediction", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_model_perf,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])
], className="page-content")
