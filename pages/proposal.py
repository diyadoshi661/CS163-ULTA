import dash
from dash import html

# Register the page
dash.register_page(__name__, path='/proposal', name='Project Proposal')

layout = html.Div(
    style={"backgroundColor": "#FFF9F4", "padding": "30px"},
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": "#bfa2db",
                "padding": "20px",
                "textAlign": "center",
                "borderRadius": "10px",
                "marginBottom": "30px"
            },
            children=[
                html.H1("Project Proposal", style={"color": "white", "fontFamily": "Georgia, serif"})
            ]
        ),

        # Project Summary
        html.Div([
            html.H2("Project Summary", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.P("""
                This project analyzes Ulta Beauty product reviews to uncover insights through sentiment analysis, bias detection,
                pricing strategy shifts, and explainable machine learning. By combining NLP, visual analytics, and predictive modeling,
                we aim to support more transparent product evaluation and smarter retail decisions.
            """, style={"fontSize": "18px", "color": "#333", "marginTop": "10px"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Sources
        html.Div([
            html.H2("Data Sources", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Ulta Website Scraping (Product Listings + Reviews)", style={"marginBottom": "8px"}),
                html.Li("Google Trends API (Consumer interest over time)", style={"marginBottom": "8px"}),
                html.Li("Allure Beauty Blog Articles (Industry Insights)")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Expected Major Findings
        html.Div([
            html.H2("Expected Major Findings", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Sentiment, bias, and sales patterns across Dermalogica reviews", style={"marginBottom": "8px"}),
                html.Li("Category pricing dynamics: HOT vs COLD categories", style={"marginBottom": "8px"}),
                html.Li("Explainable ML model for star rating prediction using SHAP")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Preprocessing Steps
        html.Div([
            html.H2("Preprocessing Steps", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Data Collection and Cleaning", style={"marginBottom": "8px"}),
                html.Li("Scaling and Normalization", style={"marginBottom": "8px"}),
                html.Li("Feature Engineering and Selection")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Analysis and Algorithms
        html.Div([
            html.H2("Data Analysis and Algorithms", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Sentiment and Subjectivity Scoring", style={"marginBottom": "8px"}),
                html.Li("Cluster Analysis for Bias Detection", style={"marginBottom": "8px"}),
                html.Li("XGBoost Modeling and SHAP Analysis")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Visualization Plan
        html.Div([
            html.H2("Data Visualization Plan", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Violin plots for price and rating distributions", style={"marginBottom": "8px"}),
                html.Li("Radar charts for review behavior clustering", style={"marginBottom": "8px"}),
                html.Li("SHAP summary plots for explainability")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "40px"}),
    ]
)
