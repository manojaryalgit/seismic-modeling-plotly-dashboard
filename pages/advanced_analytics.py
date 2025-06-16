import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
from functools import lru_cache
from scipy import stats, signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

dash.register_page(__name__, name='Advanced Analytics', icon='microscope')

@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv("data/Final-after-feature-enginnering.csv")
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    return df

df = load_data()

# --- Advanced Analytics Functions ---

def calculate_fourier_analysis(series, sampling_rate=1):
    """Perform Fourier analysis on time series data"""
    fft = np.fft.fft(series)
    freqs = np.fft.fftfreq(len(series), d=1/sampling_rate)
    
    # Only keep positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = np.abs(fft[:len(fft)//2])
    
    return positive_freqs, positive_fft

def calculate_wavelet_analysis(series):
    """Simple wavelet-like analysis using continuous wavelet transform approximation"""
    scales = np.arange(1, 32)
    coefficients = []
    
    for scale in scales:
        # Simple approximation using convolution
        kernel = signal.ricker(scale*4, scale)
        coef = np.convolve(series, kernel, mode='same')
        coefficients.append(coef)
    
    return scales, np.array(coefficients)

# --- Create Advanced Analytics Figures ---

# 1. Fractal Dimension Analysis
def calculate_box_counting_dimension(coords, max_box_size=1.0, num_sizes=20):
    """Calculate fractal dimension using box counting method"""
    box_sizes = np.logspace(-2, np.log10(max_box_size), num_sizes)
    counts = []
    
    for size in box_sizes:
        # Create grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_bins = int((x_max - x_min) / size) + 1
        y_bins = int((y_max - y_min) / size) + 1
        
        # Count occupied boxes
        hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_bins, y_bins])
        occupied_boxes = np.sum(hist > 0)
        counts.append(occupied_boxes)
    
    # Linear regression on log-log plot
    log_sizes = np.log10(box_sizes)
    log_counts = np.log10(counts)
    
    # Remove any infinite values
    valid_mask = np.isfinite(log_sizes) & np.isfinite(log_counts)
    if np.sum(valid_mask) > 1:
        slope, intercept, r_value, _, _ = stats.linregress(log_sizes[valid_mask], log_counts[valid_mask])
        fractal_dimension = -slope
    else:
        fractal_dimension = 0
        r_value = 0
    
    return box_sizes, counts, fractal_dimension, r_value

coords = df[['LAT', 'LON']].dropna().values
box_sizes, box_counts, fractal_dim, r_value = calculate_box_counting_dimension(coords)

fig_fractal = go.Figure()

fig_fractal.add_trace(go.Scatter(
    x=np.log10(box_sizes),
    y=np.log10(box_counts),
    mode='markers+lines',
    line=dict(color='#58a6ff', width=2),
    marker=dict(size=8),
    name='Box Counting',
    hovertemplate='<b>Log Box Size:</b> %{x:.3f}<br>' +
                  '<b>Log Count:</b> %{y:.3f}<extra></extra>'
))

fig_fractal.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(28, 33, 40, 0.95)',
    height=400,
    title={'text': f'Fractal Dimension Analysis (D = {fractal_dim:.3f}, R² = {r_value**2:.3f})', 
           'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
    margin=dict(t=80, b=40, l=60, r=40),
    xaxis_title='Log₁₀(Box Size)',
    yaxis_title='Log₁₀(Box Count)'
)

# 2. Spectral Analysis of Earthquake Sequences
df_sorted = df.sort_values('Date and Time').dropna(subset=['MAG'])
if len(df_sorted) > 10:
    # Resample to regular intervals
    df_resampled = df_sorted.set_index('Date and Time').resample('D')['MAG'].mean().fillna(0)
    
    if len(df_resampled) > 10:
        freqs, fft_vals = calculate_fourier_analysis(df_resampled.values)
        
        fig_spectral = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Time Series of Daily Average Magnitude', 'Frequency Spectrum'),
            vertical_spacing=0.12
        )
        
        # Time series
        fig_spectral.add_trace(
            go.Scatter(
                x=df_resampled.index,
                y=df_resampled.values,
                mode='lines',
                line=dict(color='#58a6ff', width=2),
                name='Daily Avg Magnitude',
                hovertemplate='<b>Date:</b> %{x}<br>' +
                              '<b>Magnitude:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Frequency spectrum
        fig_spectral.add_trace(
            go.Scatter(
                x=freqs[1:],  # Skip DC component
                y=fft_vals[1:],
                mode='lines',
                line=dict(color='#2ea043', width=2),
                name='Power Spectrum',
                hovertemplate='<b>Frequency:</b> %{x:.4f} cycles/day<br>' +
                              '<b>Power:</b> %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_spectral.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            height=600,
            title={'text': 'Spectral Analysis of Earthquake Activity', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
            margin=dict(t=80, b=40, l=60, r=40),
            showlegend=False
        )
    else:
        fig_spectral = go.Figure().add_annotation(
            text="Insufficient data for spectral analysis",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#f0f6fc')
        )
        fig_spectral.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)
else:
    fig_spectral = go.Figure().add_annotation(
        text="Insufficient data for spectral analysis",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_spectral.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)

# 3. Network Analysis of Earthquake Connections
def create_earthquake_network(df, distance_threshold=50, magnitude_threshold=4.5):
    """Create network based on spatial and temporal proximity"""
    # Filter significant earthquakes
    significant_eq = df[df['MAG'] >= magnitude_threshold].copy()
    
    if len(significant_eq) < 2:
        return None
    
    # Calculate distances between earthquakes
    G = nx.Graph()
    
    for i, eq1 in significant_eq.iterrows():
        G.add_node(i, magnitude=eq1['MAG'], lat=eq1['LAT'], lon=eq1['LON'], 
                  date=eq1['Date and Time'])
        
        for j, eq2 in significant_eq.iterrows():
            if i != j:
                # Calculate distance (simplified)
                dist = np.sqrt((eq1['LAT'] - eq2['LAT'])**2 + (eq1['LON'] - eq2['LON'])**2) * 111  # km
                time_diff = abs((eq1['Date and Time'] - eq2['Date and Time']).total_seconds() / (24*3600))  # days
                
                if dist < distance_threshold and time_diff < 30:  # Within 30 days
                    G.add_edge(i, j, distance=dist, time_diff=time_diff)
    
    return G

G = create_earthquake_network(df)

if G is not None and len(G.nodes()) > 0:
    # Create network visualization
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract node and edge information
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(
            size=[G.nodes[node]['magnitude'] * 3 for node in G.nodes()],
            color=[G.nodes[node]['magnitude'] for node in G.nodes()],
            colorscale='Viridis',
            colorbar=dict(title="Magnitude"),
            line=dict(width=2, color='white')
        ),
        text=[f"M{G.nodes[node]['magnitude']:.1f}" for node in G.nodes()],
        textposition="middle center",
        textfont=dict(size=8, color='white'),
        name='Earthquakes',
        hovertemplate='<b>Magnitude:</b> %{marker.color:.1f}<br>' +
                      '<b>Connections:</b> %{text}<extra></extra>'
    )
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.extend([go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                    mode='lines', line=dict(width=1, color='rgba(125, 125, 125, 0.5)'),
                                    hoverinfo='none', showlegend=False)])
    
    fig_network = go.Figure(data=edge_trace + [node_trace])
    fig_network.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=500,
        title={'text': f'Earthquake Network Analysis ({len(G.nodes())} nodes, {len(G.edges())} edges)', 
               'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        margin=dict(t=80, b=40, l=40, r=40),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
else:
    fig_network = go.Figure().add_annotation(
        text="Insufficient data for network analysis",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_network.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)

# 4. Self-Organizing Map Visualization (simplified)
def create_som_analysis(data, grid_size=(10, 10)):
    """Simplified SOM-like analysis for earthquake clustering"""
    if len(data) < 10:
        return None
    
    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Simple grid-based clustering
    x_min, x_max = data_scaled[:, 0].min(), data_scaled[:, 0].max()
    y_min, y_max = data_scaled[:, 1].min(), data_scaled[:, 1].max()
    
    x_grid = np.linspace(x_min, x_max, grid_size[0])
    y_grid = np.linspace(y_min, y_max, grid_size[1])
    
    # Assign points to grid
    grid_counts = np.zeros(grid_size)
    
    for point in data_scaled:
        x_idx = np.argmin(np.abs(x_grid - point[0]))
        y_idx = np.argmin(np.abs(y_grid - point[1]))
        grid_counts[y_idx, x_idx] += 1
    
    return x_grid, y_grid, grid_counts

# Use LAT, LON, MAG for SOM analysis
som_data = df[['LAT', 'LON', 'MAG']].dropna().values
som_result = create_som_analysis(som_data)

if som_result is not None:
    x_grid, y_grid, grid_counts = som_result
    
    fig_som = go.Figure(data=go.Heatmap(
        z=grid_counts,
        x=np.arange(len(x_grid)),
        y=np.arange(len(y_grid)),
        colorscale='Viridis',
        hovertemplate='<b>Grid X:</b> %{x}<br>' +
                      '<b>Grid Y:</b> %{y}<br>' +
                      '<b>Count:</b> %{z}<extra></extra>'
    ))
    
    fig_som.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=400,
        title={'text': 'Self-Organizing Map Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        margin=dict(t=80, b=40, l=60, r=40),
        xaxis_title='SOM Grid X',
        yaxis_title='SOM Grid Y'
    )
else:
    fig_som = go.Figure().add_annotation(
        text="Insufficient data for SOM analysis",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_som.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)

# 5. Anomaly Detection Analysis
def detect_anomalies(df, features=['MAG', 'DEPTH'], contamination=0.1):
    """Detect anomalies using statistical methods"""
    data = df[features].dropna()
    
    if len(data) < 10:
        return None, None
    
    # Z-score based anomaly detection
    z_scores = np.abs(stats.zscore(data))
    anomalies_zscore = (z_scores > 3).any(axis=1)
    
    # IQR based anomaly detection
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    anomalies_iqr = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
    
    return anomalies_zscore, anomalies_iqr

anomalies_z, anomalies_iqr = detect_anomalies(df)

if anomalies_z is not None:
    fig_anomalies = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Z-Score Anomaly Detection', 'IQR Anomaly Detection')
    )
    
    # Z-Score anomalies
    normal_data = df[~anomalies_z]
    anomaly_data = df[anomalies_z]
    
    fig_anomalies.add_trace(
        go.Scatter(
            x=normal_data['DEPTH'],
            y=normal_data['MAG'],
            mode='markers',
            marker=dict(color='#58a6ff', size=6, opacity=0.7),
            name='Normal',
            hovertemplate='<b>Depth:</b> %{x:.1f} km<br>' +
                          '<b>Magnitude:</b> %{y:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    if len(anomaly_data) > 0:
        fig_anomalies.add_trace(
            go.Scatter(
                x=anomaly_data['DEPTH'],
                y=anomaly_data['MAG'],
                mode='markers',
                marker=dict(color='#f85149', size=8, symbol='x'),
                name='Z-Score Anomalies',
                hovertemplate='<b>Depth:</b> %{x:.1f} km<br>' +
                              '<b>Magnitude:</b> %{y:.1f}<br>' +
                              '<b>Status:</b> Anomaly<extra></extra>'
            ),
            row=1, col=1
        )
    
    # IQR anomalies
    normal_data_iqr = df[~anomalies_iqr]
    anomaly_data_iqr = df[anomalies_iqr]
    
    fig_anomalies.add_trace(
        go.Scatter(
            x=normal_data_iqr['DEPTH'],
            y=normal_data_iqr['MAG'],
            mode='markers',
            marker=dict(color='#58a6ff', size=6, opacity=0.7),
            name='Normal IQR',
            showlegend=False,
            hovertemplate='<b>Depth:</b> %{x:.1f} km<br>' +
                          '<b>Magnitude:</b> %{y:.1f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    if len(anomaly_data_iqr) > 0:
        fig_anomalies.add_trace(
            go.Scatter(
                x=anomaly_data_iqr['DEPTH'],
                y=anomaly_data_iqr['MAG'],
                mode='markers',
                marker=dict(color='#f85149', size=8, symbol='diamond'),
                name='IQR Anomalies',
                hovertemplate='<b>Depth:</b> %{x:.1f} km<br>' +
                              '<b>Magnitude:</b> %{y:.1f}<br>' +
                              '<b>Status:</b> Anomaly<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig_anomalies.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=500,
        title={'text': 'Anomaly Detection Analysis', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        margin=dict(t=80, b=40, l=60, r=40)
    )
else:
    fig_anomalies = go.Figure().add_annotation(
        text="Insufficient data for anomaly detection",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_anomalies.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)

# 6. Phase Space Reconstruction
def phase_space_reconstruction(series, delay=1, dimension=3):
    """Create phase space reconstruction for time series analysis"""
    if len(series) < dimension * delay:
        return None
    
    N = len(series) - (dimension - 1) * delay
    phase_space = np.zeros((N, dimension))
    
    for i in range(dimension):
        phase_space[:, i] = series[i * delay:N + i * delay]
    
    return phase_space

# Use magnitude time series for phase space reconstruction
mag_series = df_sorted['MAG'].values
phase_space = phase_space_reconstruction(mag_series, delay=5, dimension=3)

if phase_space is not None:
    fig_phase = go.Figure(data=[go.Scatter3d(
        x=phase_space[:, 0],
        y=phase_space[:, 1],
        z=phase_space[:, 2],
        mode='markers+lines',
        marker=dict(
            size=3,
            color=np.arange(len(phase_space)),
            colorscale='Viridis',
            colorbar=dict(title="Time Index")
        ),
        line=dict(color='rgba(88, 166, 255, 0.3)', width=2),
        name='Phase Space Trajectory',
        hovertemplate='<b>M(t):</b> %{x:.2f}<br>' +
                      '<b>M(t+τ):</b> %{y:.2f}<br>' +
                      '<b>M(t+2τ):</b> %{z:.2f}<extra></extra>'
    )])
    
    fig_phase.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(28, 33, 40, 0.95)',
        height=600,
        title={'text': 'Phase Space Reconstruction of Magnitude Time Series', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
        scene=dict(
            xaxis_title='M(t)',
            yaxis_title='M(t+τ)',
            zaxis_title='M(t+2τ)',
            bgcolor='rgba(28, 33, 40, 0.95)'
        ),
        margin=dict(t=80, b=40, l=40, r=40)
    )
else:
    fig_phase = go.Figure().add_annotation(
        text="Insufficient data for phase space reconstruction",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color='#f0f6fc')
    )
    fig_phase.update_layout(template="plotly_dark", paper_bgcolor='rgba(28, 33, 40, 0.95)', height=400)

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-microscope me-3 text-accent"),
            "Advanced Analytics"
        ], className="mb-0 fw-bold"),
        html.P("Cutting-edge analytical techniques including fractal analysis, network theory, and nonlinear dynamics", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # Fractal and Complexity Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-infinity me-2 text-accent"),
            "Fractal and Complexity Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-snowflake me-2"),
                            "Fractal Dimension Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Box-counting fractal dimension analysis revealing spatial complexity of earthquake distribution", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_fractal,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-project-diagram me-2"),
                            "Self-Organizing Map"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("SOM-based clustering revealing hidden patterns in earthquake characteristics", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_som,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=6)
        ], className="mb-5")
    ]),
    
    # Signal Processing and Spectral Analysis Section
    html.Div([
        html.H2([
            html.I(className="fas fa-wave-square me-2 text-accent"),
            "Signal Processing and Spectral Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-area me-2"),
                            "Spectral Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Fourier analysis of earthquake time series revealing periodic patterns and frequency content", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_spectral,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Network and Graph Theory Section
    html.Div([
        html.H2([
            html.I(className="fas fa-share-alt me-2 text-accent"),
            "Network and Graph Theory Analysis"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-network-wired me-2"),
                            "Earthquake Network Analysis"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Network graph showing connections between earthquakes based on spatiotemporal proximity", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_network,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Nonlinear Dynamics Section
    html.Div([
        html.H2([
            html.I(className="fas fa-tornado me-2 text-accent"),
            "Nonlinear Dynamics and Chaos Theory"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-cube me-2"),
                            "Phase Space Reconstruction"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("3D phase space reconstruction revealing attractor dynamics in earthquake magnitude time series", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_phase,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ], className="mb-5")
    ]),
    
    # Anomaly Detection Section
    html.Div([
        html.H2([
            html.I(className="fas fa-search me-2 text-accent"),
            "Advanced Anomaly Detection"
        ], className="section-title mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-exclamation-circle me-2"),
                            "Multi-Method Anomaly Detection"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.P("Comparative anomaly detection using Z-score and IQR methods to identify unusual earthquake events", 
                               className="text-muted mb-3"),
                        dcc.Graph(
                            figure=fig_anomalies,
                            config={'displayModeBar': True, 'displaylogo': False}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ])
    ])
], className="page-content")
