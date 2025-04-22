import dash
from dash import html

dash.register_page(__name__, path='/objective', name='Project Objective')

layout = html.Div([
    html.H2("Project Objectives"),
    html.Ul([
        html.Li("Understand user sentiment across skincare product reviews."),
        html.Li("Benchmark brands and categories based on ratings, price, and review count."),
        html.Li("Identify biased or extreme language using NLP techniques."),
        html.Li("Develop explainable ML models to understand what drives ratings."),
    ])
])

