from dash import Dash, html, dcc, Output, Input
import dash

app = Dash(__name__, use_pages=True)

# Define custom page order
custom_order = [
    "Home Page",
    "Project Proposal",
    "Major Findings",
    "Analytical Methods",
    "Project Objective"
]

# Splash screen layout
def splash_screen():
    return html.Div(
        id="splash",
        style={
            "position": "fixed",
            "top": "0", "left": "0", "right": "0", "bottom": "0",
            "background-image": "url('/assets/makeup-scaled.jpg')",
            "background-size": "cover",
            "background-position": "center",
            "zIndex": "9999",
            "display": "flex",
            "flex-direction": "column",
            "justify-content": "center",
            "align-items": "center",
            "color": "white",
            "text-shadow": "1px 1px 6px rgba(0,0,0,0.8)",
            "text-align": "center",
            "font-family": "Comic Sans MS, cursive, sans-serif"
        },
        children=[
            html.H1("Fleure Beauty's", style={"font-size": "4rem", "margin": "0.2em"}),
            html.H2("Sunkissed Collection", style={"font-size": "2.5rem", "margin": "0.1em"}),
            dcc.Interval(id="splash-timer", interval=3_000, n_intervals=0)
        ]
    )

# Main app layout
def full_layout():
    sorted_pages = sorted(dash.page_registry.values(), key=lambda p: custom_order.index(p['name']) if p['name'] in custom_order else 99)
    return html.Div([
        html.H1("Ulta Skincare Review Dashboard", style={"textAlign": "center", "font-family": "Comic Sans MS, cursive, sans-serif"}),
        html.Hr(),
        html.Div([
            html.Span([
                dcc.Link(f"{page['name']} ", href=page["relative_path"], style={"padding": "0 10px", "text-decoration": "none", "color": "#444", "font-family": "Comic Sans MS, cursive, sans-serif"}),
                html.Span("|", style={"color": "#bbb"})
            ]) for page in sorted_pages
        ], style={"textAlign": "center", "font-size": "1.1rem"}),
        html.Hr(),
        dash.page_container
    ])

# Show splash screen first
app.layout = html.Div(id='root-layout', children=splash_screen())

# Swap to main layout after timer
@app.callback(
    Output('root-layout', 'children'),
    Input('splash-timer', 'n_intervals')
)
def update_to_main_layout(n):
    if n > 0:
        return full_layout()
    return dash.no_update

if __name__ == "__main__":
    app.run(debug=True)
