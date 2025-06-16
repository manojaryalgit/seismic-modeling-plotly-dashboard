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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, name='Statistical Analysis', icon='chart-area')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    return df

df = load_data()

# Select numerical features for analysis
numerical_features = ['MAG', 'DEPTH', 'LAT', 'LON', 'MAG_Rolling_Mean30_1', 'MAG_Rolling_Mean7_1', 
                     'a_value_1', 'b_value_1', 'sqrt_Energy_Release', 'MAG_Cumulative_Mean_1',
                     'days_since_last_eq_gt4', 'days_since_last_eq_gt5']

# Filter features that exist in the dataframe
available_features = [col for col in numerical_features if col in df.columns]
df_numerical = df[available_features].dropna()

# --- Create Enhanced Statistical Figures ---

# 1. Correlation Matrix Heatmap
correlation_matrix = df_numerical.corr()

# Create custom colorscale
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=np.round(correlation_matrix.values, 2),
    texttemplate="%{text}",
    textfont={"size": 10},
    hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>' +
                  'Correlation: %{z:.3f}<extra></extra>'
))

fig_corr.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Feature Correlation Matrix', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=100, r=40),
    xaxis={'side': 'bottom', 'tickangle': 45},
    yaxis={'side': 'left'}
)

# 2. Distribution Analysis
fig_distributions = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Magnitude Distribution', 'Depth Distribution', 'Latitude Distribution',
                   'Longitude Distribution', 'b-value Distribution', 'Energy Release Distribution'),
    specs=[[{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}],
           [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]]
)

features_to_plot = ['MAG', 'DEPTH', 'LAT', 'LON', 'b_value_1', 'sqrt_Energy_Release']
colors = ['#58a6ff', '#2ea043', '#d29922', '#f85149', '#8b5cf6', '#fb7185']

for i, feature in enumerate(features_to_plot):
    if feature in df.columns:
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        # Add histogram
        fig_distributions.add_trace(
            go.Histogram(
                x=df[feature].dropna(),
                nbinsx=30,
                marker_color=colors[i],
                opacity=0.8,
                name=feature,
                hovertemplate=f'<b>{feature}:</b> %{{x:.3f}}<br>' +
                              '<b>Count:</b> %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add normal distribution overlay if data is available
        if len(df[feature].dropna()) > 0:
            data = df[feature].dropna()
            mu, sigma = stats.norm.fit(data)
            x_norm = np.linspace(data.min(), data.max(), 100)
            y_norm = len(data) * (x_norm[1] - x_norm[0]) * stats.norm.pdf(x_norm, mu, sigma)
            
            fig_distributions.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    line=dict(color='white', width=2, dash='dash'),
                    name=f'Normal Fit ({feature})',
                    showlegend=False,
                    hovertemplate=f'<b>Normal Fit</b><br>' +
                                  f'<b>{feature}:</b> %{{x:.3f}}<br>' +
                                  f'<b>Density:</b> %{{y:.2f}}<extra></extra>'
                ),
                row=row, col=col
            )

fig_distributions.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Statistical Distributions with Normal Fits', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    showlegend=False
)

# 3. Q-Q Plots for Normality Testing
fig_qq = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Magnitude Q-Q Plot', 'Depth Q-Q Plot', 'b-value Q-Q Plot', 'Energy Release Q-Q Plot')
)

qq_features = ['MAG', 'DEPTH', 'b_value_1', 'sqrt_Energy_Release']
qq_colors = ['#58a6ff', '#2ea043', '#8b5cf6', '#fb7185']

for i, feature in enumerate(qq_features):
    if feature in df.columns:
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        data = df[feature].dropna()
        if len(data) > 0:
            # Calculate theoretical quantiles
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
            sample_quantiles = np.sort(data)
            
            # Add Q-Q scatter plot
            fig_qq.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    marker=dict(color=qq_colors[i], size=4, opacity=0.7),
                    name=f'{feature} Q-Q',
                    hovertemplate=f'<b>Theoretical:</b> %{{x:.3f}}<br>' +
                                  f'<b>Sample:</b> %{{y:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add reference line
            min_q, max_q = min(theoretical_quantiles), max(theoretical_quantiles)
            fig_qq.add_trace(
                go.Scatter(
                    x=[min_q, max_q],
                    y=[min_q * np.std(data) + np.mean(data), max_q * np.std(data) + np.mean(data)],
                    mode='lines',
                    line=dict(color='white', width=2, dash='dash'),
                    name=f'{feature} Reference',
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )

fig_qq.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Q-Q Plots for Normality Assessment', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    showlegend=False
)

# 4. Principal Component Analysis
if len(df_numerical) > 0 and len(df_numerical.columns) > 2:
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    
    # Create PCA plots
    fig_pca = make_subplots(
        rows=1, cols=2,
        subplot_titles=('PCA Scatter Plot (PC1 vs PC2)', 'Explained Variance Ratio'),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # PCA scatter plot
    fig_pca.add_trace(
        go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=dict(
                color=df_numerical['MAG'] if 'MAG' in df_numerical.columns else 'blue',
                colorscale='Viridis',
                size=6,
                opacity=0.7,
                colorbar=dict(title="Magnitude")
            ),
            name='Earthquakes',
            hovertemplate='<b>PC1:</b> %{x:.3f}<br>' +
                          '<b>PC2:</b> %{y:.3f}<br>' +
                          '<b>Magnitude:</b> %{marker.color:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Explained variance
    fig_pca.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            marker_color='#58a6ff',
            name='Explained Variance',
            hovertemplate='<b>Component:</b> %{x}<br>' +
                          '<b>Explained Variance:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig_pca.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=500,
        title={'text': 'Principal Component Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        margin=dict(t=80, b=40, l=60, r=40),
        showlegend=False
    )
else:
    fig_pca = go.Figure()
    fig_pca.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=500,
        title={'text': 'Principal Component Analysis - Insufficient Data', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        margin=dict(t=80, b=40, l=60, r=40)
    )

# 5. Statistical Summary Table
summary_stats = df_numerical.describe().round(3)

fig_summary = go.Figure(data=[go.Table(
    header=dict(
        values=['Statistic'] + list(summary_stats.columns),
        fill_color='#1c2128',
        font=dict(color='#f0f6fc', size=12),
        align='left'
    ),
    cells=dict(
        values=[summary_stats.index] + [summary_stats[col] for col in summary_stats.columns],
        fill_color='rgba(28, 33, 40, 0.95)',
        font=dict(color='#f0f6fc', size=11),
        align='left'
    )
)])

fig_summary.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Descriptive Statistics Summary', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=40, r=40)
)

# 6. Box Plots for Outlier Detection
fig_boxplots = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Magnitude Boxplot', 'Depth Boxplot', 'b-value Boxplot',
                   'Energy Release Boxplot', 'Rolling Mean 30 Boxplot', 'Days Since Last EQ Boxplot')
)

box_features = ['MAG', 'DEPTH', 'b_value_1', 'sqrt_Energy_Release', 'MAG_Rolling_Mean30_1', 'days_since_last_eq_gt4']
box_colors = ['#58a6ff', '#2ea043', '#8b5cf6', '#fb7185', '#d29922', '#f85149']

for i, feature in enumerate(box_features):
    if feature in df.columns:
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        fig_boxplots.add_trace(
            go.Box(
                y=df[feature].dropna(),
                name=feature,
                marker_color=box_colors[i],
                showlegend=False,
                hovertemplate=f'<b>{feature}</b><br>' +
                              '<b>Value:</b> %{y:.3f}<extra></extra>'
            ),
            row=row, col=col
        )

fig_boxplots.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Box Plots for Outlier Detection', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40)
)

# 7. Skewness and Kurtosis Analysis
skew_kurt_data = []
for feature in available_features:
    if feature in df.columns:
        data = df[feature].dropna()
        if len(data) > 0:
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            skew_kurt_data.append({
                'Feature': feature,
                'Skewness': skewness,
                'Kurtosis': kurtosis
            })

skew_kurt_df = pd.DataFrame(skew_kurt_data)

fig_skew_kurt = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Skewness Analysis', 'Kurtosis Analysis')
)

if not skew_kurt_df.empty:
    # Skewness
    fig_skew_kurt.add_trace(
        go.Bar(
            x=skew_kurt_df['Feature'],
            y=skew_kurt_df['Skewness'],
            marker_color=['#2ea043' if x > -0.5 and x < 0.5 else '#f85149' for x in skew_kurt_df['Skewness']],
            name='Skewness',
            hovertemplate='<b>Feature:</b> %{x}<br>' +
                          '<b>Skewness:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Kurtosis
    fig_skew_kurt.add_trace(
        go.Bar(
            x=skew_kurt_df['Feature'],
            y=skew_kurt_df['Kurtosis'],
            marker_color=['#2ea043' if x > -2 and x < 2 else '#f85149' for x in skew_kurt_df['Kurtosis']],
            name='Kurtosis',
            hovertemplate='<b>Feature:</b> %{x}<br>' +
                          '<b>Kurtosis:</b> %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )

fig_skew_kurt.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    title={'text': 'Skewness and Kurtosis Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    showlegend=False,
    xaxis=dict(tickangle=45),
    xaxis2=dict(tickangle=45)
)

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-area me-3 text-accent"),
            "Statistical Analysis"
        ], className="mb-0 fw-bold"),
        html.P("Advanced statistical analysis including correlations, distributions, normality tests, and dimensionality reduction", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Correlation Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-project-diagram me-2 text-accent"),
            "Correlation Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-th me-2"),
                            "Feature Correlation Matrix"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Correlation heatmap showing relationships between numerical features", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_corr,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Distribution Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-chart-bar me-2 text-accent"),
            "Distribution Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-line me-2"),
                            "Statistical Distributions"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Histograms with normal distribution overlays for key features", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_distributions,
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
                            html.I(className="fas fa-square-root-alt me-2"),
                            "Normality Assessment"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Q-Q plots to assess normality of key feature distributions", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_qq,
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
                            html.I(className="fas fa-chart-simple me-2"),
                            "Shape Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Skewness and kurtosis analysis to understand distribution shapes", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_skew_kurt,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Outlier Detection Section
    html.Div([
        html.H2([
            html.I(className="fas fa-search me-2 text-accent"),
            "Outlier Detection"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-box me-2"),
                            "Box Plot Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Box plots showing quartiles, medians, and outliers for key features", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_boxplots,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Dimensionality Reduction Section
    html.Div([
        html.H2([
            html.I(className="fas fa-compress me-2 text-accent"),
            "Dimensionality Reduction"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-vector-square me-2"),
                            "Principal Component Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("PCA analysis showing principal components and explained variance", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_pca,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Summary Statistics Section
    html.Div([
        html.H2([
            html.I(className="fas fa-table me-2 text-accent"),
            "Summary Statistics"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-list-alt me-2"),
                            "Descriptive Statistics"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Comprehensive statistical summary including mean, std, quartiles, and extremes", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_summary,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])
], className="page-content")
