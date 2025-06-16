import dash
import dash_bootstrap_components as dbc
from dash import html, Output, Input, callback

# Use a professional dark Bootstrap theme
app = dash.Dash(__name__, use_pages=True, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    "https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap"
])
server = app.server

# Define the navbar with improved styling and organized navigation
navbar = dbc.Navbar(
    dbc.Container([
        # Brand/Logo section
        dbc.NavbarBrand(
            [
                html.I(className="fas fa-globe-americas me-2"),
                "Nepal Earthquake Analysis"
            ], 
            href="/",
            className="fw-bold"
        ),
        
        # Toggle button for mobile
        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
        
        # Navigation links organized by category
        dbc.Collapse([
            dbc.Nav([
                # Core Analysis Pages
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-home me-1"), "Overview"],
                        href="/",
                        active="exact",
                        className="nav-link-custom",
                        id="nav-overview"
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-chart-bar me-1"), "Features"],
                        href="/feature-analysis",
                        active="exact", 
                        className="nav-link-custom",
                        id="nav-features"
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-brain me-1"), "Models"],
                        href="/model-performance",
                        active="exact",
                        className="nav-link-custom", 
                        id="nav-models"
                    )
                ),
                
                # Separator
                html.Span("|", className="mx-2 text-muted"),
                
                # Spatial & Temporal Analysis
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-map-marked-alt me-1"), "Spatial"],
                        href="/spatial-analysis",
                        active="exact",
                        className="nav-link-custom",
                        id="nav-spatial"
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-clock me-1"), "Temporal"],
                        href="/temporal-analysis", 
                        active="exact",
                        className="nav-link-custom",
                        id="nav-temporal"
                    )
                ),
                
                # Separator
                html.Span("|", className="mx-2 text-muted"),
                
                # Advanced Analysis
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-chart-area me-1"), "Statistics"],
                        href="/statistical-analysis",
                        active="exact", 
                        className="nav-link-custom",
                        id="nav-statistics"
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-exclamation-triangle me-1"), "Risk"],
                        href="/risk-assessment",
                        active="exact",
                        className="nav-link-custom",
                        id="nav-risk"
                    )
                ),
                dbc.NavItem(
                    dbc.NavLink(
                        [html.I(className="fas fa-microscope me-1"), "Advanced"],
                        href="/advanced-analytics",
                        active="exact",
                        className="nav-link-custom",
                        id="nav-advanced"
                    )
                ),
            ], className="ms-auto", navbar=True)
        ], id="navbar-collapse", navbar=True)
    ], fluid=True),
    color="dark",
    dark=True,
    className="border-bottom border-secondary",
    sticky="top"
)

# Define the main layout with improved structure
app.layout = html.Div([
    navbar,
    dbc.Container([
        # Main content area
        html.Div([
            dash.page_container
        ], className="main-content")
    ], fluid=True, className="px-4")
], className="app-container")

# Callback for navbar toggle
@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    prevent_initial_call=True,
)
def toggle_navbar_collapse(n):
    if n:
        return not dash.callback_context.inputs_list[0]["value"]
    return False

if __name__ == '__main__':
    app.run(debug=True, port=8051)

