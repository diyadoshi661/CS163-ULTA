import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/', name='Home Page')
# Load example datasets (replace with your actual DataFrame loading logic)
df_old = pd.read_csv('cleaned_makeup_products.csv')
df_new = pd.read_csv('face_df.csv')

# Data Preprocessing for Visualization
old_top_brands = df_old['brand'].value_counts().nlargest(10).reset_index()
old_top_brands.columns = ['Brand', 'Product Count']
new_top_brands = df_new['Brand'].value_counts().nlargest(10).reset_index()
new_top_brands.columns = ['Brand', 'Product Count']

# Average price comparison
price_comparison = pd.DataFrame({
    'Dataset': ['Old', 'New'],
    'Average Price': [df_old['price'].mean(), df_new['Price'].mean()]
})

# Average review comparison
review_comparison = pd.DataFrame({
    'Dataset': ['Old', 'New'],
    'Average Reviews': [df_old['num_reviews'].mean(), df_new['Reviews'].mean()]
})

# Visualizations
fig_top_old = px.bar(old_top_brands, x='Brand', y='Product Count', title='Top 10 Hottest Brands (Old Data)')
fig_top_new = px.bar(new_top_brands, x='Brand', y='Product Count', title='Top 10 Hottest Brands (New Data)')
fig_price = px.bar(price_comparison, x='Dataset', y='Average Price', title='Average Price Comparison')
fig_reviews = px.bar(review_comparison, x='Dataset', y='Average Reviews', title='Average Review Count Comparison')

fig_violin_price = px.violin(
    df_new, 
    x="Category", 
    y="Price", 
    box=True, 
    points="all", 
    color="Category", 
    title="Price Distribution Across Categories"
)
fig_violin_price.update_layout(
    xaxis_title="Category",
    yaxis_title="Price ($)",
    xaxis_tickangle=45,
    showlegend=False
)

fig_violin_stars = px.violin(
    df_new, 
    x="Category", 
    y="Weighted_Rating", 
    box=True, 
    points="all", 
    color="Category", 
    title="Weighted Rating Distribution Across Categories"
)
fig_violin_stars.update_layout(
    xaxis_title="Category",
    yaxis_title="Weighted Rating",
    xaxis_tickangle=45,
    showlegend=False
)
# Dash Layout
layout = html.Div(
    style={"backgroundColor": "#FFF9F4"},
    children=[
        # Header section
        html.Div(
            style={
                "background-image": "url('/assets/bg.png')",
                "background-size": "cover",
                "background-position": "center",
                "height": "50vh",
                "display": "flex",
                "flex-direction": "column",
                "justify-content": "center",
                "align-items": "flex-start",
                "color": "#FFF9F4",
                "text-shadow": "1px 1px 6px rgba(0,0,0,0.7)",
                "padding-left": "60px",
                "font-family": "Georgia, serif"
            },
            children=[
                html.H1("Ulta Beauty", style={
                    "font-size": "4rem",
                    "font-weight": "bold",
                    "margin": "0"
                }),
                html.H2("Data Analysis", style={
                    "font-size": "2rem",
                    "font-weight": "normal",
                    "margin-top": "0.3rem"
                }),
            ]
        ),
        html.Div(
            children=[
                html.P(
                    "This project analyzes Ulta Beauty product reviews to uncover insights through sentiment analysis, "
                    "bias detection, pricing strategy shifts, and explainable machine learning. "
                    "By combining NLP, visual analytics, and predictive modeling, we aim to support more transparent "
                    "product evaluation and smarter retail decisions.",
                    style={
                        "maxWidth": "80%",
                        "margin": "30px auto 10px auto",
                        "textAlign": "center",
                        "fontSize": "1.1rem",
                        "color": "#333",
                        "font-family": "Georgia, serif"
                    }
                )
            ]
        ),
        # Visualization section
        html.Div([
            html.H3("Insights Dashboard", style={"textAlign": "center", "marginTop": "40px"}),

            html.Div([
                html.Div([dcc.Graph(figure=fig_top_old)], style={"width": "48%", "display": "inline-block", "padding": "1%"}),
                html.Div([dcc.Graph(figure=fig_top_new)], style={"width": "48%", "display": "inline-block", "padding": "1%"})
            ]),

            html.Div([
                html.Div([dcc.Graph(figure=fig_price)], style={"width": "48%", "display": "inline-block", "padding": "1%"}),
                html.Div([dcc.Graph(figure=fig_reviews)], style={"width": "48%", "display": "inline-block", "padding": "1%"})
            ])
        ], style={"padding": "0 40px"}),
        
        html.Div([
            dcc.Graph(figure=fig_violin_price),
            dcc.Graph(figure=fig_violin_stars)
        ], style={"padding": "0 40px"})
    ]
)
