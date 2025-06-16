import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import joblib
from pages.feature_analysis import load_data
from sklearn.preprocessing import StandardScaler


dash.register_page(__name__, name='Model Performance', icon='brain')

# --- Load Models and Test Data ---
# This assumes you have run the snippet to save your models and y_test
try:
    y_test = joblib.load('models/y_test.joblib')
    # Manually enter results from your notebook for display
    # We create a dictionary to hold all relevant info for each model
    model_results = {
        'ridge': {
            'model': None,
            'r2': 0.0312,
            'mae': 0.3497,
            'features': ['a/b_1', 'MAG_Rolling_Mean30_1', 'MAG_Rolling_Mean7_1', 'MAG_Cumulative_Mean_1'] # Simplified for LR
        },
        'rf': {
            'model': None,
            'r2': 0.4785,
            'mae': 0.2452,
            'features': ['a/b_1', 'MAG_Cumulative_Mean_1', 'MAG_Rolling_Mean30_1', 'MAG_Rolling_Mean7_1',
                         'sqrt_Energy_Release', 'b/a_1', 'DEPTH', 'days_since_last_eq_gt4']
        },
        'xgb': {
            'model': None,
            'r2': 0.4866,
            'mae': 0.2443,
            'features': ['a/b_1', 'MAG_Rolling_Mean30_1', 'MAG_Cumulative_Mean_1', 'MAG_Rolling_Mean7_1',
                         'sqrt_Energy_Release', 'b/a_1', 'days_since_last_eq_gt4', 'DEPTH']
        },
        'cat': {
            'model': None,
            'r2': 0.4857,
            'mae': 0.2464,
            'features': ['a/b_1', 'MAG_Rolling_Mean30_1', 'MAG_Cumulative_Mean_1', 'MAG_Rolling_Mean7_1',
                         'sqrt_Energy_Release', 'b/a_1', 'days_since_last_eq_gt4', 'DEPTH']
        }
    }
    
    # Try to load each model individually
    models_to_load = [
        ('ridge', 'models/ridge_model.joblib'),
        ('rf', 'models/rf_model.joblib'),
        ('xgb', 'models/xgb_model.joblib'),
        ('cat', 'models/cat_model.joblib')
    ]
    
    for model_name, model_path in models_to_load:
        try:
            model_results[model_name]['model'] = joblib.load(model_path)
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Could not load {model_name} model: {e}")
            model_results[model_name]['model'] = None
    
    MODELS_LOADED = True
    print("Model loading completed")
    
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    MODELS_LOADED = False
except Exception as e:
    print(f"Error during model loading: {e}")
    MODELS_LOADED = False

# --- Layout ---
if not MODELS_LOADED:
    layout = html.Div([
        dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "Models not found. Please run the training and saving script first."
        ], color="danger", className="text-center")
    ])
else:
    layout = html.Div([
        # Page header
        html.Div([
            html.H1([
                html.I(className="fas fa-brain me-3 text-accent"),
                "Model Performance Analysis"
            ], className="mb-0 fw-bold"),
            html.P("Compare and analyze machine learning model performance for earthquake prediction", 
                   className="lead text-muted mb-4")
        ], className="page-header mb-5"),
        
        # Control Panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Model Selection"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        html.Label("Choose a Model:", className="form-label fw-semibold mb-3"),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[
                                {'label': '🔗 Ridge Regression', 'value': 'ridge'},
                                {'label': '🌳 Random Forest', 'value': 'rf'},
                                {'label': '⚡ XGBoost', 'value': 'xgb'},
                                {'label': '🐱 CatBoost', 'value': 'cat'},
                            ],
                            value='xgb',
                            clearable=False,
                            className="model-dropdown",
                            style={'z-index': '1000'}
                        )
                    ], className="p-4")
                ])
            ], width=12, className="mb-4")
        ]),
        
        # KPI Cards Row
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-line fa-2x text-accent mb-3"),
                        html.H3("R² Score", className="kpi-label"),
                        html.Div(id='r2-card', className="kpi-value mb-0")
                    ], className="text-center")
                ], className="kpi-card h-100")
            ], width=6),
            
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-bullseye fa-2x text-warning mb-3"),
                        html.H3("Mean Absolute Error", className="kpi-label"),
                        html.Div(id='mae-card', className="kpi-value mb-0")
                    ], className="text-center")
                ], className="kpi-card h-100")
            ], width=6),
        ], className="mb-5"),
        
        # Visualization Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-chart-area me-2"),
                            "Actual vs Predicted Values"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='actual-vs-predicted-plot',
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '500px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm mb-4")
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H4([
                            html.I(className="fas fa-star me-2"),
                            "Feature Importance"
                        ], className="card-title mb-0")
                    ], className="bg-transparent border-0 pb-0"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='feature-importance-plot',
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '500px', 'width': '100%'}
                        )
                    ], className="p-4")
                ], className="shadow-sm")
            ], width=12)
        ])
    ], className="page-content")

# --- The "Master" Callback for this Page ---
@callback(
    Output('r2-card', 'children'),
    Output('mae-card', 'children'),
    Output('actual-vs-predicted-plot', 'figure'),
    Output('feature-importance-plot', 'figure'),
    Input('model-selector', 'value')
)
def update_model_outputs(selected_model):
    if not MODELS_LOADED:
        return "", "", {}, {}

    # Get results from the pre-computed dictionary
    results = model_results[selected_model]
    model = results['model']
    
    # Format the KPI values
    r2_value = html.H2(f"{results['r2']:.4f}", className="mb-0 fw-bold text-accent")
    mae_value = html.H2(f"{results['mae']:.4f}", className="mb-0 fw-bold text-warning")
    
    # If model is not loaded, show static metrics only
    if model is None:
        # Create a placeholder prediction plot
        fig_predictions = go.Figure()
        fig_predictions.add_annotation(
            text=f"Model {selected_model.upper()} not available<br>Showing performance metrics only",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#8b949e')
        )
        fig_predictions.update_layout(
            title={
                'text': f"Model Performance: {selected_model.upper()}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            plot_bgcolor='rgba(28, 33, 40, 0.95)'
        )
        
        # Create a placeholder feature importance plot
        fig_importance = go.Figure()
        fig_importance.add_annotation(
            text="Feature importance not available<br>Model not loaded",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#8b949e')
        )
        fig_importance.update_layout(
            title={
                'text': "Feature Importance Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            plot_bgcolor='rgba(28, 33, 40, 0.95)'
        )
        
        return r2_value, mae_value, fig_predictions, fig_importance
    
    # Get predictions by loading the appropriate test set
    try:
        full_df = load_data()
        # Simplified feature selection - in a real scenario, you'd save the specific X_test for each model
        if selected_model == 'cat' and hasattr(model, 'feature_names_'):
            features = model.feature_names_
        elif hasattr(model, 'feature_names_in_'):
            features = model.feature_names_in_
        else:
            features = results['features']

        X = full_df[features]
        y = full_df['MAG']
        split_index = int(0.8 * len(X))
        ss = StandardScaler()
        ss.fit(X[:split_index])
        
        X_test = ss.transform(X[split_index:])
        y_test_subset = y[split_index:]

        y_pred = model.predict(X_test)

        # Create Actual vs. Predicted Plot with enhanced styling
        fig_predictions = go.Figure()
        
        # Add actual values
        fig_predictions.add_trace(go.Scatter(
            x=y_test_subset.index, 
            y=y_test_subset, 
            mode='lines+markers', 
            name='Actual', 
            line=dict(color='#58a6ff', width=3), 
            marker=dict(size=6, color='#58a6ff', line=dict(width=1, color='#f0f6fc'))
        ))
        
        # Add predicted values
        fig_predictions.add_trace(go.Scatter(
            x=y_test_subset.index, 
            y=y_pred, 
            mode='lines+markers', 
            name='Predicted', 
            line=dict(color='#f85149', dash='dash', width=3),
            marker=dict(size=4, color='#f85149', line=dict(width=1, color='#f0f6fc'))
        ))
        
        fig_predictions.update_layout(
            title={
                'text': f"Model Performance: {selected_model.upper()}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            template="plotly_dark",
            xaxis_title="Time Index",
            yaxis_title="Magnitude",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            plot_bgcolor='rgba(28, 33, 40, 0.95)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color='#f0f6fc')
            ),
            margin=dict(t=60, b=40, l=60, r=40),
            height=460,
            xaxis=dict(
                gridcolor='rgba(48, 54, 61, 0.5)',
                color='#f0f6fc'
            ),
            yaxis=dict(
                gridcolor='rgba(48, 54, 61, 0.5)',
                color='#f0f6fc'
            )
        )
        
        # Create Feature Importance Plot with enhanced styling
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)  # Changed to ascending for better visualization
            
            fig_importance = px.bar(
                importance_df.tail(10), 
                x='importance', 
                y='feature', 
                orientation='h', 
                title=f"Top 10 Feature Importances - {selected_model.upper()}", 
                template="plotly_dark",
                color='importance',
                color_continuous_scale=[[0, '#2ea043'], [0.5, '#d29922'], [1, '#58a6ff']]
            )
            
            fig_importance.update_layout(
                title={
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#f0f6fc'}
                },
                xaxis_title="Importance Score",
                yaxis_title="Features",
                paper_bgcolor='rgba(28, 33, 40, 0.95)',
                plot_bgcolor='rgba(28, 33, 40, 0.95)',
                margin=dict(t=60, b=40, l=150, r=40),
                height=460,
                xaxis=dict(
                    gridcolor='rgba(48, 54, 61, 0.5)',
                    color='#f0f6fc'
                ),
                yaxis=dict(
                    gridcolor='rgba(48, 54, 61, 0.5)',
                    color='#f0f6fc'
                ),
                coloraxis_colorbar=dict(
                    title="Importance",
                    title_font_color='#f0f6fc',
                    tickfont_color='#f0f6fc'
                )
            )
            
            fig_importance.update_traces(
                texttemplate='%{x:.3f}',
                textposition='outside',
                textfont_color='#f0f6fc',
                marker_line_color='#f0f6fc',
                marker_line_width=1
            )
            
        else:
            # For models like Ridge that don't have feature_importances_
            fig_importance = go.Figure()
            fig_importance.add_annotation(
                text="Feature importance not available<br>for this model type",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color='#8b949e')
            )
            fig_importance.update_layout(
                title={
                    'text': "Feature Importance Analysis",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18, 'color': '#f0f6fc'}
                },
                template="plotly_dark",
                paper_bgcolor='rgba(28, 33, 40, 0.95)',
                plot_bgcolor='rgba(28, 33, 40, 0.95)'
            )

        return r2_value, mae_value, fig_predictions, fig_importance
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        # Create error plots
        fig_predictions = go.Figure()
        fig_predictions.add_annotation(
            text=f"Error generating predictions<br>{str(e)[:50]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#f85149')
        )
        fig_predictions.update_layout(
            title={'text': "Prediction Error", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            plot_bgcolor='rgba(28, 33, 40, 0.95)'
        )
        
        fig_importance = go.Figure()
        fig_importance.add_annotation(
            text="Error loading feature importance",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#f85149')
        )
        fig_importance.update_layout(
            title={'text': "Feature Importance Error", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': '#f0f6fc'}},
            template="plotly_dark",
            paper_bgcolor='rgba(28, 33, 40, 0.95)',
            plot_bgcolor='rgba(28, 33, 40, 0.95)'
        )
        
        return r2_value, mae_value, fig_predictions, fig_importance
