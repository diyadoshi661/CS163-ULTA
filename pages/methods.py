import dash
from dash import html

dash.register_page(__name__, path='/methods', name='Methodology')

layout = html.Div(
    style={"backgroundColor": "#FFF9F4", "padding": "30px", "font-family": "Georgia, serif"},
    children=[
        html.Div(
            style={"textAlign": "center", "padding": "20px", "fontSize": "3rem", "fontWeight": "bold", "color": "#2F4F4F"},
            children="Analytical Methods"
        ),

        # Cards
        html.Div([
            html.Div([
                html.H2("Data Preprocessing", style={"color": "#4B0082", "marginBottom": "10px"}),
                html.P("Cleaning reviews, handling missing data, label encoding categorical features.", 
                       style={"color": "#333", "fontSize": "1.2rem"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "500px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H2("Sentiment Analysis", style={"color": "#4B0082", "marginBottom": "10px"}),
                html.P("VADER/TextBlob or DistilBERT", style={"color": "#333", "fontSize": "1.2rem"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "500px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H2("Keyword Extraction", style={"color": "#4B0082", "marginBottom": "10px"}),
                html.P("TF-IDF, KeyBERT", style={"color": "#333", "fontSize": "1.2rem"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "500px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H2("XGBoost Regression", style={"color": "#4B0082", "marginBottom": "10px"}),
                html.P("Predicting star ratings", style={"color": "#333", "fontSize": "1.2rem"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "500px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

            html.Div([
                html.H2("SHAP Explainability", style={"color": "#4B0082", "marginBottom": "10px"}),
                html.P("Visualizing feature impacts", style={"color": "#333", "fontSize": "1.2rem"})
            ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "500px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center"})
    ]
)
