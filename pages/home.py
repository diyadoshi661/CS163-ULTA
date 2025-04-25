import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px

dash.register_page(__name__, path='/home', name='Home Page')
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

# Dash Layout
layout = html.Div([
    html.Div(
        style={
            "background-image": "url('/assets/makeup-scaled.jpg')",
            "background-size": "cover",
            "background-position": "center",
            "height": "50vh",
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
            html.H1("Fleure Beauty's", style={"font-size": "4rem", "margin": "0.25em"}),
            html.H2("Ulta Beauty Data Analysis", style={"font-size": "2.5rem", "margin": "0.1em"})
        ]
    ),
    html.Div([
        html.H3("Insights Dashboard", style={"text-align": "center", "marginTop": "40px"}),
        dcc.Graph(figure=fig_top_old),
        dcc.Graph(figure=fig_top_new),
    ], style={"padding": "0 40px"})
])
