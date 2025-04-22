from dash import Dash, html, dcc, Output, Input
import dash

app = Dash(__name__, use_pages=True)

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
            "text-align": "center"
        },
        children=[
            html.H1("Fleure Beauty's", style={"font-size": "4rem", "margin": "0.2em"}),
            html.H2("Sunkissed Collection", style={"font-size": "2.5rem", "margin": "0.1em"}),
            # ⏱️ Changed from 30s to 7s
            dcc.Interval(id="splash-timer", interval=3_000, n_intervals=0)
        ]
    )

# Main app layout (no image, just title, nav, and pages)
def full_layout():
    return html.Div([
        html.H1("Ulta Skincare Review Dashboard", style={"textAlign": "center"}),
        html.Hr(),
        html.Div([
            dcc.Link(f"{page['name']} | ", href=page["relative_path"])
            for page in dash.page_registry.values()
        ], style={"textAlign": "center"}),
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
