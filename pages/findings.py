import dash
from dash import html

dash.register_page(__name__, path='/findings', name='Major Findings')

layout = html.Div([
    html.H2("Major Findings"),
    html.P("This section highlights key insights discovered from the Ulta skincare dataset."),
    html.Ul([
        html.Li("Certain premium brands have consistently high ratings regardless of price."),
        html.Li("High review counts do not always correlate with high ratings — brand loyalty may play a role."),
        html.Li("Some products with lower star ratings show very positive sentiment in text — indicating potential rating bias."),
        html.Li("Price and category combinations reveal clusters of underpriced high-quality items."),
    ]),
    html.P("Visualizations to support these findings will be added here."),
])

