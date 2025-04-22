import dash
from dash import html

dash.register_page(__name__, path='/', name='Home')

layout = html.Div([
    html.Div(
        style={
            "background-image": "url('/assets/makeup-scaled.jpg')",
            "background-size": "cover",
            "background-position": "center",
            "height": "90vh",
            "display": "flex",
            "flex-direction": "column",
            "justify-content": "center",
            "align-items": "center",
            "color": "white",
            "text-shadow": "1px 1px 6px rgba(0,0,0,0.7)",
            "text-align": "center",
            "padding": "0 20px"
        },
        children=[
            html.H1("Fleure Beauty's", style={
                "font-size": "4rem",
                "margin": "0.25em"
            }),
            html.H2("Sunkissed Collection", style={
                "font-size": "2.5rem",
                "margin": "0.1em"
            })
        ]
    )
])
