import dash
from dash import html

dash.register_page(__name__, path='/proposal', name='Project Proposal')

layout = html.Div(
    style={
        'position': 'relative',
        'minHeight': '100vh',
        'backgroundImage': "url('/assets/background.png')",
        'backgroundSize': 'cover',
        'backgroundPosition': 'center',
        'backgroundRepeat': 'no-repeat',
        'backgroundColor': 'rgba(255, 255, 255, 0.5)',  # White overlay for transparency
        'backgroundBlendMode': 'overlay',
        'padding': '2rem',
        'fontFamily': "'Comic Sans MS', cursive, sans-serif"
    },
    children=[
        html.H1("Project Proposal", style={'textAlign': 'center', 'color': '#222'}),
        html.H2("Project Summary", style={'color': '#B22222'}),
        html.P("""
        With an emphasis on spotting possible biases, fake reviews, and the impact of marketing tactics like influencer promotions 
        and exclusivity, this project seeks to assess the dependability and credibility of consumer evaluations on Ulta's platform...
        """, style={'color': '#333'}),

        html.H2("Broader Impacts", style={'color': '#228B22'}),
        html.Ul([
            html.Li("Increased Transparency and Trust"),
            html.Li("Ethical Marketing Practices"),
            html.Li("Consumer Empowerment"),
        ], style={'color': '#444'}),

        html.H2("Data Sources", style={'color': '#4169E1'}),
        html.Ul([
            html.Li(" Ulta Website"),
            html.Li("Google Trends API"),
            html.Li("Allure Beauty Blog"),
        ], style={'color': '#444'}),

        html.H2("Expected Major Findings", style={'color': '#8B008B'}),
        html.Ul([
            html.Li(" Seasonal Influence on Reviews"),
            html.Li(" Marketing & Review Bias"),
            html.Li("Sentiment Analysis"),
        ], style={'color': '#444'}),
    ]
)
