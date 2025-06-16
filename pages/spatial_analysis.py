import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
from functools import lru_cache
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, name='Spatial Analysis', icon='map-marked-alt')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    return df

df = load_data()

# Create spatial clustering
def perform_spatial_clustering():
    coords = df[['LAT', 'LON']].dropna()
    
    # K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_clean = df.dropna(subset=['LAT', 'LON']).copy()
    df_clean['cluster_kmeans'] = kmeans.fit_predict(coords)
    
    # DBSCAN clustering for density-based clustering
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    df_clean['cluster_dbscan'] = dbscan.fit_predict(coords)
    
    return df_clean

df_clustered = perform_spatial_clustering()

# --- Create Enhanced Spatial Figures ---

# 1. Geographic Distribution with Magnitude
fig_geo_mag = px.scatter_mapbox(
    df, 
    lat='LAT', 
    lon='LON', 
    color='MAG',
    size='MAG',
    hover_data=['DEPTH', 'Date and Time'],
    color_continuous_scale='Viridis',
    size_max=20,
    zoom=6,
    center=dict(lat=28.3949, lon=84.1240),
    title='Earthquake Distribution by Magnitude',
    mapbox_style='open-street-map'
)
fig_geo_mag.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    margin=dict(t=60, b=40, l=40, r=40),
    title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}}
)

# 2. Depth Distribution Analysis
fig_depth_analysis = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Depth vs Magnitude', 'Depth Distribution', 'Magnitude vs Latitude', 'Magnitude vs Longitude'),
    specs=[[{"type": "scatter"}, {"type": "histogram"}],
           [{"type": "scatter"}, {"type": "scatter"}]]
)

# Depth vs Magnitude scatter
fig_depth_analysis.add_trace(
    go.Scatter(
        x=df['DEPTH'], 
        y=df['MAG'],
        mode='markers',
        marker=dict(
            color=df['MAG'],
            colorscale='Viridis',
            size=8,
            opacity=0.7
        ),
        name='Earthquakes',
        text=df['Date and Time'].dt.strftime('%Y-%m-%d'),
        hovertemplate='<b>Depth:</b> %{x:.1f} km<br>' +
                      '<b>Magnitude:</b> %{y:.1f}<br>' +
                      '<b>Date:</b> %{text}<extra></extra>'
    ),
    row=1, col=1
)

# Depth histogram
fig_depth_analysis.add_trace(
    go.Histogram(
        x=df['DEPTH'],
        nbinsx=30,
        marker_color='#58a6ff',
        opacity=0.8,
        name='Depth Distribution'
    ),
    row=1, col=2
)

# Magnitude vs Latitude
fig_depth_analysis.add_trace(
    go.Scatter(
        x=df['LAT'], 
        y=df['MAG'],
        mode='markers',
        marker=dict(
            color=df['DEPTH'],
            colorscale='Plasma',
            size=8,
            opacity=0.7,
            colorbar=dict(title="Depth (km)")
        ),
        name='Lat-Mag',
        hovertemplate='<b>Latitude:</b> %{x:.3f}°<br>' +
                      '<b>Magnitude:</b> %{y:.1f}<br>' +
                      '<b>Depth:</b> %{marker.color:.1f} km<extra></extra>'
    ),
    row=2, col=1
)

# Magnitude vs Longitude
fig_depth_analysis.add_trace(
    go.Scatter(
        x=df['LON'], 
        y=df['MAG'],
        mode='markers',
        marker=dict(
            color=df['DEPTH'],
            colorscale='Plasma',
            size=8,
            opacity=0.7
        ),
        name='Lon-Mag',
        showlegend=False,
        hovertemplate='<b>Longitude:</b> %{x:.3f}°<br>' +
                      '<b>Magnitude:</b> %{y:.1f}<br>' +
                      '<b>Depth:</b> %{marker.color:.1f} km<extra></extra>'
    ),
    row=2, col=2
)

fig_depth_analysis.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': 'Spatial-Physical Relationships', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=60)
)

# 3. Clustering Analysis
fig_clustering = make_subplots(
    rows=1, cols=2,
    subplot_titles=('K-Means Clustering', 'DBSCAN Clustering'),
    specs=[[{"type": "scatter"}, {"type": "scatter"}]]
)

# K-means clustering
colors_kmeans = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
for i, cluster in enumerate(df_clustered['cluster_kmeans'].unique()):
    cluster_data = df_clustered[df_clustered['cluster_kmeans'] == cluster]
    fig_clustering.add_trace(
        go.Scatter(
            x=cluster_data['LON'], 
            y=cluster_data['LAT'],
            mode='markers',
            marker=dict(
                color=colors_kmeans[i % len(colors_kmeans)],
                size=8,
                opacity=0.7
            ),
            name=f'K-Means Cluster {cluster}',
            hovertemplate='<b>Lat:</b> %{y:.3f}°<br>' +
                          '<b>Lon:</b> %{x:.3f}°<br>' +
                          '<b>Cluster:</b> %{fullData.name}<extra></extra>'
        ),
        row=1, col=1
    )

# DBSCAN clustering
colors_dbscan = ['#e17055', '#00b894', '#0984e3', '#6c5ce7', '#fd79a8', '#fdcb6e']
for i, cluster in enumerate(df_clustered['cluster_dbscan'].unique()):
    cluster_data = df_clustered[df_clustered['cluster_dbscan'] == cluster]
    cluster_name = 'Noise' if cluster == -1 else f'DBSCAN Cluster {cluster}'
    color = '#2d3436' if cluster == -1 else colors_dbscan[i % len(colors_dbscan)]
    
    fig_clustering.add_trace(
        go.Scatter(
            x=cluster_data['LON'], 
            y=cluster_data['LAT'],
            mode='markers',
            marker=dict(
                color=color,
                size=6 if cluster == -1 else 8,
                opacity=0.4 if cluster == -1 else 0.7
            ),
            name=cluster_name,
            hovertemplate='<b>Lat:</b> %{y:.3f}°<br>' +
                          '<b>Lon:</b> %{x:.3f}°<br>' +
                          '<b>Cluster:</b> %{fullData.name}<extra></extra>'
        ),
        row=1, col=2
    )

fig_clustering.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    title={'text': 'Spatial Clustering Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=60)
)

# 4. Density Analysis
fig_density = px.density_mapbox(
    df, 
    lat='LAT', 
    lon='LON', 
    z='MAG',
    radius=20,
    center=dict(lat=28.3949, lon=84.1240), 
    zoom=6,
    mapbox_style="open-street-map",
    title='Earthquake Density Heatmap',
    color_continuous_scale='Hot'
)
fig_density.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=500,
    margin=dict(t=60, b=40, l=40, r=40),
    title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}}
)

# 5. 3D Spatial Visualization
fig_3d = go.Figure(data=[go.Scatter3d(
    x=df['LON'],
    y=df['LAT'],
    z=df['DEPTH'],
    mode='markers',
    marker=dict(
        size=df['MAG'] * 2,
        color=df['MAG'],
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title="Magnitude")
    ),
    text=df['Date and Time'].dt.strftime('%Y-%m-%d'),
    hovertemplate='<b>Longitude:</b> %{x:.3f}°<br>' +
                  '<b>Latitude:</b> %{y:.3f}°<br>' +
                  '<b>Depth:</b> %{z:.1f} km<br>' +
                  '<b>Date:</b> %{text}<extra></extra>'
)])

fig_3d.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=600,
    title={'text': '3D Spatial Distribution', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Depth (km)',
        bgcolor='rgba(28, 33, 40, 0.95)',
        xaxis=dict(backgroundcolor='rgba(28, 33, 40, 0.95)', gridcolor='rgba(48, 54, 61, 0.5)'),
        yaxis=dict(backgroundcolor='rgba(28, 33, 40, 0.95)', gridcolor='rgba(48, 54, 61, 0.5)'),
        zaxis=dict(backgroundcolor='rgba(28, 33, 40, 0.95)', gridcolor='rgba(48, 54, 61, 0.5)')
    ),
    margin=dict(t=80, b=40, l=40, r=40)
)

# 6. Distance Analysis
# Calculate distances from Nepal's center
nepal_center_lat, nepal_center_lon = 28.3949, 84.1240
df['distance_from_center'] = np.sqrt((df['LAT'] - nepal_center_lat)**2 + (df['LON'] - nepal_center_lon)**2) * 111  # Approximate km

fig_distance = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Distance vs Magnitude', 'Distance Distribution'),
    specs=[[{"type": "scatter"}, {"type": "histogram"}]]
)

fig_distance.add_trace(
    go.Scatter(
        x=df['distance_from_center'], 
        y=df['MAG'],
        mode='markers',
        marker=dict(
            color=df['DEPTH'],
            colorscale='Viridis',
            size=8,
            opacity=0.7,
            colorbar=dict(title="Depth (km)")
        ),
        name='Distance-Magnitude',
        hovertemplate='<b>Distance:</b> %{x:.1f} km<br>' +
                      '<b>Magnitude:</b> %{y:.1f}<br>' +
                      '<b>Depth:</b> %{marker.color:.1f} km<extra></extra>'
    ),
    row=1, col=1
)

fig_distance.add_trace(
    go.Histogram(
        x=df['distance_from_center'],
        nbinsx=25,
        marker_color='#2ea043',
        opacity=0.8,
        name='Distance Distribution'
    ),
    row=1, col=2
)

fig_distance.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': 'Distance from Nepal Center Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=60)
)

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-map-marked-alt me-3 text-accent"),
            "Spatial Analysis"
        ], className="mb-0 fw-bold"),
        html.P("Comprehensive spatial analysis of earthquake patterns, clustering, and geographic relationships", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Geographic Distribution Section
    html.Div([
        html.H2([
            html.I(className="fas fa-globe me-2 text-accent"),
            "Geographic Distribution"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-map-pin me-2"),
                            "Earthquake Locations by Magnitude"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Interactive map showing earthquake epicenters colored and sized by magnitude", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_geo_mag,
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
                            html.I(className="fas fa-fire me-2"),
                            "Earthquake Density Heatmap"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Density visualization showing earthquake hotspots and concentration areas", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_density,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # 3D Visualization Section
    html.Div([
        html.H2([
            html.I(className="fas fa-cube me-2 text-accent"),
            "3D Spatial Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-mountain me-2"),
                            "3D Earthquake Distribution"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Three-dimensional view of earthquake locations including depth information", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_3d,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Physical Relationships Section
    html.Div([
        html.H2([
            html.I(className="fas fa-ruler-combined me-2 text-accent"),
            "Spatial-Physical Relationships"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-scatter me-2"),
                            "Depth and Magnitude Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Relationships between earthquake depth, magnitude, and geographic coordinates", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_depth_analysis,
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
                            html.I(className="fas fa-ruler me-2"),
                            "Distance Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Analysis of earthquake patterns relative to Nepal's geographic center", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_distance,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Clustering Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-project-diagram me-2 text-accent"),
            "Clustering Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-sitemap me-2"),
                            "Spatial Clustering"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("K-means and DBSCAN clustering to identify earthquake zones and patterns", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_clustering,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])
], className="page-content")
