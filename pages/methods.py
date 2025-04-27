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

        html.Div([
            html.P(
                "In this project, data preprocessing and feature engineering played a foundational role. "
                "We began by scraping Ulta Beauty product data and user reviews, ensuring the collection of relevant information. "
                "Missing values were carefully handled to preserve data integrity, and full review texts were created by merging separate title and comment fields into cohesive, analyzable feedback. "
                "To better capture user behavior and language patterns, we engineered several new features, including sentiment polarity, subjectivity scores, review length (both character and word counts), "
                "punctuation usage patterns (such as excessive exclamation marks), and unique word ratio, which measures lexical diversity in a review."
            )
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),

        # Box 2 - NLP
        html.Div([
            html.P(
                "For Natural Language Processing (NLP) tasks, we focused on understanding the emotional tone and reliability of reviews. "
                "Sentiment analysis was conducted using the TextBlob library, enabling us to compute both sentiment polarity (positive or negative tone) and subjectivity (degree of opinionated language). "
                "Reviews with a subjectivity score above 0.6 were labeled as 'biased,' recognizing that highly emotional reviews might distort product perceptions. "
                "To further categorize review types, we applied KMeans clustering based on linguistic features such as review length, subjectivity, and verified buyer status, helping differentiate between genuine user feedback and potentially exaggerated or low-quality reviews."
            )
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),

        # Box 3 - Predictive Modeling
        html.Div([
            html.P(
                "In the predictive modeling phase, we trained an XGBoost regression model to predict star ratings using engineered features like price, brand encoding, review count patterns, and recommendation ratios. "
                "The model achieved strong predictive performance, with root mean squared error (RMSE) equals 0.2305 and mean absolute error (MAE) equals 0.1575, validating the strength of the chosen features. "
                "To ensure model transparency, we utilized SHAP (SHapley Additive Explanations) to visualize how individual features influenced the model's predictions, revealing that factors like review volume, price, and brand affiliation significantly impacted predicted star ratings."
            )
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),

        # Box 4 - Data Visualization
        html.Div([
            html.P(
                "Lastly, data visualization was employed to bring these insights to life. "
                "We created violin plots to depict the distribution of prices and weighted ratings across different product categories, radar charts to illustrate distinct clusters of review behavior, and SHAP summary plots to explain how features contributed to model decisions. "
                "Additionally, we produced pricing evolution bar charts to monitor how brand strategies shifted over time, identifying brands that adopted premium positioning or competitive price-cutting tactics."
            )
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),


        html.H3("Price Change Distribution"),
        dcc.Graph(figure=fig_price_change),

        html.H3("Brand Pricing Strategy Shifts"),
        dcc.Graph(figure=fig_brand_trends),

        html.H3("Category Dynamics: Growth and Ratings"),
        dcc.Graph(figure=fig_category_trends),

    ])