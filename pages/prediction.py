import dash
from dash import html

dash.register_page(__name__, path='/predictions', name='Predictions')

layout = html.Div(
    style={"backgroundColor": "#FFF9F4", "padding": "30px", "font-family": "Georgia, serif"},
    children=[
        # Title
        html.Div(
    style={
        "backgroundColor": "#BFA2DB",   # ✨ Soft purple strip
        "padding": "20px",
        "textAlign": "center",
        "borderRadius": "10px",
        "marginBottom": "30px",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
    },
    children=html.H1(
        "Predictions",
        style={
            "fontSize": "3rem",
            "fontWeight": "bold",
            "color": "white",
            "margin": "0",
            "font-family": "Georgia, serif"
        }
    )
),


        # Project Summary Card
        html.Div([
            html.H2("Project Summary", style={"color": "#4B0082", "marginBottom": "10px"}),
            html.P("This section demonstrates how machine learning models are applied to predict ratings "
                   "and identify key factors influencing star scores.",
                   style={"color": "#333", "fontSize": "1.2rem"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "600px", "margin": "20px auto", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

        # Data Sources Card
        html.Div([
            html.H2("Data Sources", style={"color": "#4B0082", "marginBottom": "10px"}),
            html.Ul([
                html.Li("XGBoost: Regression model for predicting star ratings", style={"fontSize": "1.2rem"}),
                html.Li("SHAP: Explaining what features drive favorable ratings", style={"fontSize": "1.2rem"})
            ])
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "600px", "margin": "20px auto", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

        # Expected Major Findings Card
        html.Div([
            html.H2("Expected Major Findings", style={"color": "#4B0082", "marginBottom": "10px"}),
            html.Ul([
                html.Li("XGBoost: Regression model for predicting star ratings", style={"fontSize": "1.2rem"})
            ])
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "600px", "margin": "20px auto", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),
    ]
)
