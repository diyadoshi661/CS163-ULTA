import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


dash.register_page(__name__, path='/methods', name='Methodology')

# Load and process the datasets
df_old = pd.read_csv("cleaned_makeup_products.csv")
df_new = pd.read_csv("face_df.csv")

# Clean and match
df_old['product_name'] = df_old['product_name'].astype(str).str.lower().str.strip()
df_old['brand'] = df_old['brand'].astype(str).str.lower().str.strip()
df_new['Product Name'] = df_new['Product Name'].astype(str).str.lower().str.strip()
df_new['Brand'] = df_new['Brand'].astype(str).str.lower().str.strip()
df_old['match_key'] = df_old['brand'] + ' ' + df_old['product_name']
df_new['match_key'] = df_new['Brand'] + ' ' + df_new['Product Name']

import difflib

def get_best_match(key, candidates, threshold=0.9):
    match = difflib.get_close_matches(key, candidates, n=1, cutoff=threshold)
    return match[0] if match else None

new_keys = df_new['match_key'].tolist()
df_old['matched_key'] = df_old['match_key'].apply(lambda x: get_best_match(x, new_keys))
df_merged = df_old.merge(df_new, left_on='matched_key', right_on='match_key', suffixes=('_old', '_new'))

# Price change analysis
df_merged['price_change'] = df_merged['Price'] - df_merged['price']
df_merged['price_trend'] = df_merged['price_change'].apply(
    lambda x: 'Increased' if x > 0 else ('Decreased' if x < 0 else 'Unchanged')
)

# Price change distribution plot
fig_price_change = px.histogram(
    df_merged, 
    x='price_change', 
    nbins=20, 
    title="Price Change Distribution (New vs Old)",
    labels={'price_change': 'Price Change ($)'},
)
fig_price_change.add_vline(x=0, line_dash="dash", line_color="red")

# Brand strategy analysis
brand_summary = df_merged.groupby('brand').agg(
    avg_price_old=('price', 'mean'),
    avg_price_new=('Price', 'mean')
)
brand_summary['price_change'] = brand_summary['avg_price_new'] - brand_summary['avg_price_old']
brand_summary = brand_summary[brand_summary['avg_price_old'] > 0]
brand_summary['pct_change'] = ((brand_summary['avg_price_new'] - brand_summary['avg_price_old']) / brand_summary['avg_price_old']) * 100

def classify_positioning(pct):
    if pct >= 5:
        return 'Premium Move'
    elif pct <= -5:
        return 'Competitive Move'
    else:
        return 'Stable Pricing'

brand_summary['positioning'] = brand_summary['pct_change'].apply(classify_positioning)
brand_summary_sorted = brand_summary.sort_values(by='pct_change').reset_index()

fig_brand_trends = px.bar(
    brand_summary_sorted, 
    x='pct_change', 
    y='brand', 
    color='positioning',
    orientation='h',
    title="Brand Pricing Strategy Shift",
    labels={'pct_change': '% Price Change', 'brand': 'Brand'},
    color_discrete_map={'Premium Move':'#FF7F0E', 'Competitive Move':'#1F77B4', 'Stable Pricing':'#2CA02C'}
)
fig_brand_trends.add_vline(x=0, line_dash="dash", line_color="gray")

# Category trend analysis
old_stats = df_old.groupby('category').agg(
    products_old=('product_name', 'count'),
    avg_rating_old=('average_rating', 'mean'),
    total_reviews_old=('num_reviews', 'sum'),
    avg_price_old=('price', 'mean')
).dropna()

new_stats = df_new.groupby('Category').agg(
    products_new=('Product Name', 'count'),
    avg_rating_new=('Stars', 'mean'),
    total_reviews_new=('Reviews', 'sum'),
    avg_price_new=('Price', 'mean')
).dropna()

category_trend = pd.merge(old_stats, new_stats, left_index=True, right_index=True)
category_trend['product_growth'] = category_trend['products_new'] - category_trend['products_old']
category_trend['rating_change'] = category_trend['avg_rating_new'] - category_trend['avg_rating_old']
category_trend['review_growth'] = category_trend['total_reviews_new'] - category_trend['total_reviews_old']

def categorize(row):
    if row['product_growth'] > 0 and row['rating_change'] > 0 and row['review_growth'] > 0:
        return 'HOT'
    elif row['product_growth'] < 0 and row['rating_change'] < 0 and row['review_growth'] < 0:
        return 'COLD'
    else:
        return 'MIXED'

category_trend['category_status'] = category_trend.apply(categorize, axis=1)

fig_category_trends = px.scatter(
    category_trend, 
    x='product_growth', 
    y='rating_change', 
    size=category_trend['review_growth'].abs(), 
    color='category_status',
    hover_name=category_trend.index,
    title="Category Trends: Product Growth vs. Rating Change",
    labels={'product_growth': 'Product Growth', 'rating_change': 'Rating Change'},
)
fig_category_trends.add_vline(x=0, line_dash="dash", line_color="gray")
fig_category_trends.add_hline(y=0, line_dash="dash", line_color="gray")


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

        ], style={"display": "flex", "flexDirection": "column", "alignItems": "center"}),
            html.Hr(),

        html.H3("Price Change Distribution"),
        dcc.Graph(figure=fig_price_change),

        html.H3("Brand Pricing Strategy Shifts"),
        dcc.Graph(figure=fig_brand_trends),

        html.H3("Category Dynamics: Growth and Ratings"),
        dcc.Graph(figure=fig_category_trends),

    ])