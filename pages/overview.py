import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
from functools import lru_cache
import folium
from folium import plugins
from branca.colormap import LinearColormap
import json
from dash.dependencies import State
import tempfile
import os
from folium.plugins import HeatMap

# Register this page with Dash
dash.register_page(__name__, path='/', name='Seismic Overview', icon='home')

# --- Caching function to load data ---
@lru_cache(maxsize=None)
def load_data():
    try:
        df = pd.read_csv("data/Final-after-feature-enginnering.csv")
        
        # Validate required columns exist
        required_columns = ['Date and Time', 'LAT', 'LON', 'MAG', 'DEPTH']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns {missing_columns} in the dataset")
            # Create a minimal dataframe with required columns
            df = pd.DataFrame({
                'Date and Time': [pd.Timestamp.now()],
                'LAT': [28.0],
                'LON': [84.0],
                'MAG': [4.0],
                'DEPTH': [10.0]
            })
        
        # Convert date column
        df['Date and Time'] = pd.to_datetime(df['Date and Time'], errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['Date and Time'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['LAT', 'LON', 'MAG', 'DEPTH']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid coordinates or magnitude
        df = df.dropna(subset=['LAT', 'LON', 'MAG'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return a minimal sample dataset
        return pd.DataFrame({
            'Date and Time': [pd.Timestamp.now() - pd.Timedelta(days=i) for i in range(10)],
            'LAT': [28.0 + i*0.1 for i in range(10)],
            'LON': [84.0 + i*0.1 for i in range(10)],
            'MAG': [4.0 + i*0.1 for i in range(10)],
            'DEPTH': [10.0 + i for i in range(10)]
        })

df = load_data()

def create_folium_map(filtered_df):
    """Create a Folium map with earthquake data"""
    # Create a map centered on Nepal
    m = folium.Map(
        location=[28.3949, 84.1240],
        zoom_start=7,
        tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/">CARTO</a>',
        control_scale=True
    )

    # Calculate bounds from data
    sw = [filtered_df['LAT'].min(), filtered_df['LON'].min()]  # Southwest corner
    ne = [filtered_df['LAT'].max(), filtered_df['LON'].max()]  # Northeast corner
    
    # Add custom button for fitting bounds
    fit_bounds_button = """
        <div class='leaflet-control-container'>
            <div class='leaflet-top leaflet-left'>
                <div class='leaflet-control leaflet-bar' style='margin-top: 85px;'>
                    <a href='#' id='fitBounds' 
                        title='Fit map to all data'
                        style='
                            display: block;
                            width: 26px;
                            height: 26px;
                            line-height: 26px;
                            color: #666;
                            background: white;
                            font-size: 18px;
                            text-align: center;
                            text-decoration: none;
                            border-radius: 4px;
                            cursor: pointer;'
                        onclick='fitToBounds(); return false;'>
                        <i class="fas fa-compress-arrows-alt"></i>
                    </a>
                </div>
            </div>
        </div>

        <script>
        function fitToBounds() {
            try {
                var map = document.querySelector('#map')._leaflet;
                if (!map) {
                    map = document.getElementsByClassName('folium-map')[0]._leaflet_map;
                }
                var bounds = [[%s[0], %s[1]], [%s[0], %s[1]]];
                map.fitBounds(bounds, {
                    padding: [50, 50],
                    maxZoom: 10,
                    animate: true
                });
            } catch (e) {
                console.error('Error fitting bounds:', e);
            }
        }
        
        // Add the function to window object to ensure it's accessible
        window.fitToBounds = fitToBounds;
        
        // Wait for map to be ready
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(fitToBounds, 1000);
        });
        </script>
    """ % (sw, sw, ne, ne)
    
    # Add Font Awesome for the button icon
    m.get_root().header.add_child(folium.Element(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">'
    ))
    
    # Add the button
    m.get_root().html.add_child(folium.Element(fit_bounds_button))
    
    # Add fullscreen control first
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Set initial bounds
    m.fit_bounds([[sw[0], sw[1]], [ne[0], ne[1]]], padding=(50, 50))

    # Create a feature group for markers
    markers = folium.FeatureGroup(name='Earthquakes')

    # Add markers with detailed hover information
    for idx, row in filtered_df.iterrows():
        # Create detailed tooltip content
        tooltip = f"""
        <div style='font-family: Montserrat, sans-serif; 
                    background-color: rgba(0, 0, 0, 0.8); 
                    color: white; 
                    padding: 8px; 
                    border-radius: 4px;
                    font-size: 12px;'>
            <b>Magnitude:</b> {row['MAG']:.1f}<br>
            <b>Depth:</b> {row['DEPTH']:.1f} km<br>
            <b>Date:</b> {row['Date and Time'].strftime('%Y-%m-%d')}<br>
            <b>Time:</b> {row['Date and Time'].strftime('%H:%M:%S')}<br>
            <b>Location:</b> {row['LAT']:.3f}°N, {row['LON']:.3f}°E
        </div>
        """
        
        # Add invisible circle marker with hover info
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=3,
            weight=0,
            fill=True,
            fillColor='transparent',
            fillOpacity=0,
            tooltip=folium.Tooltip(tooltip, sticky=True),  # Make tooltip sticky so it's easier to read
            popup=None  # Remove popup since all info is in tooltip
        ).add_to(markers)

    # Create heatmap layer
    heat_data = [[row['LAT'], row['LON'], 1] for _, row in filtered_df.iterrows()]
    
    # Create gradient with more green levels and red only at top
    gradient = {
        0.1: 'rgba(0, 255, 0, 0)',      # Transparent
        0.1: 'rgba(0, 255, 0, 0.5)',    # Light green
        0.2: 'rgba(0, 255, 0, 0.7)',    # Medium green
        0.3: 'rgba(150, 255, 0, 0.8)',  # Yellow-green
        0.4: 'rgba(255, 255, 0, 0.85)', # Yellow
        0.5: 'rgba(255, 165, 0, 0.9)',  # Orange
        0.9: 'rgba(255, 0, 0, 0.95)',   # Red
        1.0: 'rgba(139, 0, 0, 1.0)'     # Dark red
    }
    
    # Add heatmap layer
    HeatMap(
        heat_data,
        min_opacity=0,
        max_zoom=18,
        radius=8,
        blur=1,
        gradient=gradient,
        max_val=None
    ).add_to(m)

    # Add markers on top of heatmap
    markers.add_to(m)

    # Create legend for magnitude scale
    min_mag = filtered_df['MAG'].min()
    max_mag = filtered_df['MAG'].max()
    
    # Add magnitude scale control with matching colors
    colormap = LinearColormap(
        colors=['#00ff00', '#96ff00', '#ffff00', '#ffa500', '#ff0000', '#8b0000'],
        vmin=min_mag,
        vmax=max_mag,
        caption='Earthquake Magnitude'
    )
    m.add_child(colormap)

    # Custom CSS for clean look
    custom_style = """
    <style>
    .leaflet-container {
        background: #ffffff !important;
    }
    .leaflet-control-attribution {
        background: rgba(255, 255, 255, 0.8) !important;
        color: #333 !important;
        font-size: 10px !important;
    }
    .legend {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #ddd !important;
        border-radius: 4px !important;
        padding: 6px 8px !important;
        font-family: Arial, sans-serif !important;
        font-size: 12px !important;
        color: #333 !important;
    }
    .legend i {
        width: 18px !important;
        height: 18px !important;
        float: left !important;
        margin-right: 8px !important;
        opacity: 0.7 !important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(custom_style))

    # Add interactive features
    # Mouse position with coordinates
    plugins.MousePosition(
        position='bottomleft',
        separator=' | ',
        prefix='Coordinates:',
        num_digits=4,
        format_str='Lat: {lat}, Lon: {lng}'
    ).add_to(m)

    # Generate unique filename using timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f')
    temp_dir = tempfile.gettempdir()
    map_path = os.path.join(temp_dir, f'earthquake_map_{timestamp}.html')
    
    m.save(map_path)
    
    return map_path

# --- Page Layout ---
layout = html.Div([
    # Page header
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-area me-3 text-accent"),
            "Seismic Activity Overview"
        ], className="mb-0 fw-bold"),
        html.P("Real-time earthquake monitoring and analysis for Nepal region", 
               className="lead text-muted mb-4")
    ], className="page-header mb-5"),
    
    # KPI Cards Row
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line fa-2x text-accent mb-3"),
                    html.H3("Total Earthquakes", className="kpi-label"),
                    html.H2(id="kpi-total-quakes", className="kpi-value mb-0")
                ], className="text-center")
            ], className="kpi-card h-100")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                    html.H3("Max Magnitude", className="kpi-label"),
                    html.H2(id="kpi-max-mag", className="kpi-value mb-0")
                ], className="text-center")
            ], className="kpi-card h-100")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-clock fa-2x text-info mb-3"),
                    html.H3("Days Since M > 6", className="kpi-label"),
                    html.H2(id="kpi-days-m6", className="kpi-value mb-0")
                ], className="text-center")
            ], className="kpi-card h-100")
        ], width=3),
        
        dbc.Col([
            html.Div([
                html.Div([
                    html.I(className="fas fa-history fa-2x text-success mb-3"),
                    html.H3("Days Since M > 5", className="kpi-label"),
                    html.H2(id="kpi-days-m5", className="kpi-value mb-0")
                ], className="text-center")
            ], className="kpi-card h-100")
        ], width=3),
    ], className="mb-5"),

    # Main Map Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-map-marked-alt me-2"),
                        "Earthquake Distribution Map"
                    ], className="card-title mb-0")
                ], className="bg-transparent border-0 pb-0"),
                dbc.CardBody([
                    # Controls section
                    html.Div([
                        html.H5("Select Date Range", className="mb-4 text-accent text-center"),
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    dcc.DatePickerSingle(
                                        id='start-date-picker',
                                        min_date_allowed=df['Date and Time'].min().date(),
                                        max_date_allowed=df['Date and Time'].max().date(),
                                        initial_visible_month=df['Date and Time'].min().date(),
                                        date=df['Date and Time'].min().date(),
                                        display_format='YYYY-MM-DD',
                                        placeholder='YYYY-MM-DD',
                                        first_day_of_week=1,  # Monday
                                        className="date-picker-input",
                                        style={
                                            'width': '100%',
                                            'z-index': '1500'
                                        }
                                    )
                                ], className="date-picker-container")
                            ], xs=12, sm=5, className="mb-3 mb-sm-0"),
                            
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-arrow-right")
                                ], className="arrow-container d-flex align-items-center justify-content-center h-100")
                            ], xs=12, sm=2, className="mb-3 mb-sm-0"),
                            
                            dbc.Col([
                                html.Div([
                                    dcc.DatePickerSingle(
                                        id='end-date-picker',
                                        min_date_allowed=df['Date and Time'].min().date(),
                                        max_date_allowed=df['Date and Time'].max().date(),
                                        initial_visible_month=df['Date and Time'].max().date(),
                                        date=df['Date and Time'].max().date(),
                                        display_format='YYYY-MM-DD',
                                        placeholder='YYYY-MM-DD',
                                        first_day_of_week=1,  # Monday
                                        className="date-picker-input",
                                        style={
                                            'width': '100%',
                                            'z-index': '1500'
                                        }
                                    )
                                ], className="date-picker-container")
                            ], xs=12, sm=5)
                        ], className="align-items-center", justify="center")
                    ], className="controls-section mb-4 p-4"),
                    
                    # Map container with HTML content
                    html.Div([
                        html.Div(id='map-container', className="map-content"),
                    ], className="map-container", style={'height': '620px', 'width': '100%'})
                ], className="p-4")
            ], className="shadow-sm")
        ], width=12)
    ])
], className="page-content")

# --- Callback for interactivity ---
@callback(
    Output('end-date-picker', 'min_date_allowed'),
    Input('start-date-picker', 'date')
)
def update_end_date_min(start_date):
    """Update end date minimum based on selected start date"""
    if start_date is None:
        return df['Date and Time'].min().date()
    return start_date

@callback(
    Output('start-date-picker', 'max_date_allowed'),
    Input('end-date-picker', 'date')
)
def update_start_date_max(end_date):
    """Update start date maximum based on selected end date"""
    if end_date is None:
        return df['Date and Time'].max().date()
    return end_date

@callback(
    Output('map-container', 'children'),
    Output('kpi-total-quakes', 'children'),
    Output('kpi-max-mag', 'children'),
    Output('kpi-days-m6', 'children'),
    Output('kpi-days-m5', 'children'),
    Input('start-date-picker', 'date'),
    Input('end-date-picker', 'date')
)
def update_overview(start_date, end_date):
    try:
        # Ensure we have valid data
        if df is None or df.empty:
            return html.Div("No data available"), "0", "0.00", "N/A", "N/A"
        
        # Handle None dates with safe defaults
        if start_date is None or end_date is None:
            if not df.empty and 'Date and Time' in df.columns and df['Date and Time'].notna().any():
                start_date = df['Date and Time'].min()
                end_date = df['Date and Time'].max()
            else:
                start_date = pd.Timestamp('2020-01-01')
                end_date = pd.Timestamp.now()
        else:
            # Convert input dates safely
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
            except (ValueError, TypeError):
                start_date = df['Date and Time'].min() if not df.empty else pd.Timestamp('2020-01-01')
                end_date = df['Date and Time'].max() if not df.empty else pd.Timestamp.now()

        # Ensure start_date <= end_date
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        # Filter data safely
        try:
            date_mask = (df['Date and Time'] >= start_date) & (df['Date and Time'] <= end_date)
            filtered_df = df[date_mask].copy()
        except (KeyError, TypeError):
            filtered_df = df.copy()
        
        # Create and save the Folium map
        map_path = create_folium_map(filtered_df)
        
        # Read the map HTML content
        with open(map_path, 'r') as f:
            map_html = f.read()
        
        # Create an iframe with the map content
        map_iframe = html.Iframe(
            srcDoc=map_html,
            style={
                'width': '100%',
                'height': '600px',
                'border': 'none',
                'borderRadius': '8px',
                'backgroundColor': 'transparent'
            }
        )
        
        # Calculate KPIs
        total_quakes = len(filtered_df)
        max_mag = filtered_df['MAG'].max() if 'MAG' in filtered_df.columns and not filtered_df['MAG'].isna().all() else 0
        max_mag = float(max_mag) if not pd.isna(max_mag) else 0
        
        # Days since calculations
        days_since_m6 = "N/A"
        days_since_m5 = "N/A"
        
        try:
            if not filtered_df.empty and 'Date and Time' in filtered_df.columns and 'MAG' in filtered_df.columns:
                latest_date = filtered_df["Date and Time"].max()
                m6_quakes = filtered_df[filtered_df["MAG"] > 6]
                m5_quakes = filtered_df[filtered_df["MAG"] > 5]

                if not m6_quakes.empty:
                    days_m6 = (latest_date - m6_quakes["Date and Time"].max()).days
                    days_since_m6 = str(max(0, days_m6)) if not pd.isna(days_m6) else "N/A"
                    
                if not m5_quakes.empty:
                    days_m5 = (latest_date - m5_quakes["Date and Time"].max()).days
                    days_since_m5 = str(max(0, days_m5)) if not pd.isna(days_m5) else "N/A"
        except Exception:
            pass  # Keep default "N/A" values

        # Clean up the temporary file
        try:
            os.remove(map_path)
        except:
            pass

        return (
            map_iframe,
            f"{total_quakes:,}",
            f"{max_mag:.2f}",
            days_since_m6,
            days_since_m5
        )
        
    except Exception as e:
        print(f"Callback error: {e}")
        return html.Div(f"Error loading map: {str(e)}"), "Error", "Error", "Error", "Error"
