import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import shap
import numpy as np



dash.register_page(__name__, path='/methods', name='Methodology')

# Load and process the datasets
df_old = pd.read_csv("data/cleaned_makeup_products.csv")
df_new = pd.read_csv("data/face_df.csv")

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

# Radar chart for cluster behavior
from math import pi

# Cluster summary manually (or recompute if needed)
cluster_summary = pd.DataFrame({
    'char_count': [29.62, 346.90, 30.93],
    'word_count': [5.17, 65.39, 5.20],
    'subjectivity': [0.18, 0.57, 0.74],
    'sentiment': [-0.004, 0.254, 0.644],
    'unique_word_ratio': [0.993, 0.790, 0.992],
    'is_verified_buyer': [0.015, 0.127, 0.025]
}, index=['Cluster 0', 'Cluster 1', 'Cluster 2'])

# Normalize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cluster_normalized = pd.DataFrame(
    scaler.fit_transform(cluster_summary),
    columns=cluster_summary.columns,
    index=cluster_summary.index
)

# Radar chart prep
features = list(cluster_normalized.columns)
num_vars = len(features)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # close circle

# Plotly Radar Chart
fig_radar = go.Figure()

for cluster in cluster_normalized.index:
    values = cluster_normalized.loc[cluster].tolist()
    values += values[:1]  # repeat first value to close loop
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=features + [features[0]],
        fill='toself',
        name=cluster
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    title="Normalized Radar Chart: Cluster Behavior Comparison",
    showlegend=True
)


# Predict (optional if you want)
# preds = model.predict(X_test)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

features_to_keep = [
    'price', 
    'brand',
    'rating',
    'num_shades', 
    'num_reviews', 
    'review_star_1', 
    'review_star_2', 
    'review_star_3', 
    'review_star_4', 
    'review_star_5',
    'native_review_count', 
    'syndicated_review_count'
]

# Create a new DataFrame with only the selected features
dfxg = df_old[features_to_keep].copy()
# Prepare X and y for model
dfxg = dfxg.dropna(subset=["rating"])
dfxg['Brand_encoded'] = pd.factorize(dfxg['brand'])[0]
X = dfxg[['price', 
    'Brand_encoded',
    'num_shades', 
    'num_reviews', 
    'review_star_1', 
    'review_star_2', 
    'review_star_3', 
    'review_star_4', 
    'review_star_5',
    'native_review_count', 
    'syndicated_review_count']]
y = dfxg["rating"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBRegressor()
model.fit(X_train, y_train)

# Now model is trained and available!

# SHAP Explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

explainer = shap.TreeExplainer(model)

# 2. Compute SHAP values on the test set
shap_values = explainer.shap_values(X_test)

# 3. SHAP Summary Plot (Scatter version)
fig_shap_summary = go.Figure()
shap.summary_plot(shap_values, X_test, show=False)  # Generate but don't show (for plotly)

fig_shap_summary = px.scatter(
    x=shap_values.flatten(), 
    y=np.repeat(X_test.columns, shap_values.shape[0]),
    labels={"x": "SHAP Value", "y": "Feature"},
    title="SHAP Summary Plot (Feature Impact)",
)

# 4. SHAP Bar Plot (Mean Absolute SHAP Value per Feature)
shap_bar = np.abs(shap_values).mean(axis=0)
shap_bar_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_SHAP_Abs": shap_bar
}).sort_values(by="Mean_SHAP_Abs", ascending=True)  # Sort small to large for bar

fig_shap_bar = px.bar(
    shap_bar_df,
    x="Mean_SHAP_Abs",
    y="Feature",
    orientation="h",
    title="Mean Absolute SHAP Value by Feature",
    labels={"Mean_SHAP_Abs": "Average Impact", "Feature": "Feature"}
)

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

html.H3("Cluster Behavior Radar Chart", style={"textAlign": "center", "marginTop": "20px"}),
dcc.Graph(figure=fig_radar),        # Box 2 - NLP
html.Div([
    html.P(
        "For Natural Language Processing (NLP) tasks, we focused on understanding the emotional tone and reliability of reviews. "
        "Sentiment analysis was conducted using the TextBlob library, enabling us to compute both sentiment polarity (positive or negative tone) and subjectivity (degree of opinionated language). "
        "Reviews with a subjectivity score above 0.6 were labeled as 'biased,' recognizing that highly emotional reviews might distort product perceptions. "
        "To further categorize review types, we applied KMeans clustering based on linguistic features such as review length, subjectivity, and verified buyer status, helping differentiate between genuine user feedback and potentially exaggerated or low-quality reviews."
    )
], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),

html.H3("SHAP Summary Plot (Feature Impact)", style={"textAlign": "center", "marginTop": "30px"}),
dcc.Graph(figure=fig_shap_summary),

html.H3("Mean Absolute SHAP Value (Feature Importance)", style={"textAlign": "center", "marginTop": "30px"}),
dcc.Graph(figure=fig_shap_bar),


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
        html.Div([
            html.P(
                "By analyzing how brands adjusted their product prices over an 8-month period by matching items"
                "between the old and new datasets. We calculated the average price change per brand"
                "(except new brand doesn't have record from the old dataset) and categorized brands"
                "as having increased, decreased, or maintained their pricing.")
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),


        html.H3("Brand Pricing Strategy Shifts"),
        dcc.Graph(figure=fig_brand_trends),
        html.Div([
            html.P(
                "The insignt that we found for Some brands significantly increased average prices (e.g.,  like Juvia's place or Kylie costometics)."
                "Others decreased their prices, suggesting a more competitive strategy(e.g., Exa, revolution beauty)."
                "Several brands maintained stable pricing, indicating consistency in market positioning.(e.g., wyn beauty, morphe 2). "
                "For some products with large price fluctuations, we can selectively compare them with their rating trends and study " 
                "the relationship between them, and thereby formulate targeted price plans and the optimal selection of product supply volumes for the products."
                "Understanding pricing evolution helps applying discounting strategies, and supports strategic brand positioning decisions across product tiers.")
        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),



        html.H3("Category Dynamics: Growth and Ratings"),
        dcc.Graph(figure=fig_category_trends),
        html.Div([
            html.P(
                "We evaluated category-wise shifts by comparing changes in product count,"
                " average rating, and total reviews across time. Categories were then labeled"
                " as HOT, COLD, or MIXED based on growth trends. HOT categories showed increases"
                "in product listings, ratings, and reviews and COLD categories decline in all three areas." 
                "This analysis will provide a strategic plan for product selection, helps prioritize resource allocation."
                )

        ], style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),


    ])