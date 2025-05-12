import dash
from dash import html, dcc
import pandas as pd
import plotly.express as px
import os
from fetch import get_cleaned_makeup_products, get_face_df

df_old = get_cleaned_makeup_products()
df_new = get_face_df()


dash.register_page(__name__, path='/', name='Home')

# New brands per category
old_category_brands = df_old.groupby('category')['brand'].unique().apply(set)
new_category_brands = df_new.groupby('Category')['Brand'].unique().apply(set)

new_brands_per_category = {}
for category in new_category_brands.index:
    old_brands = old_category_brands.get(category, set())
    new_brands = new_category_brands.get(category, set())
    added_brands = new_brands - old_brands
    new_brands_per_category[category] = len(added_brands)

new_brands_df = pd.DataFrame(list(new_brands_per_category.items()), columns=['Category', 'New Brands Count'])
new_brands_df = new_brands_df.sort_values(by='New Brands Count', ascending=False)

# New products per brand
old_brand_products = df_old.groupby('brand')['product_name'].unique().apply(set)
new_brand_products = df_new.groupby('Brand')['Product Name'].unique().apply(set)

new_products_per_brand = {}
for brand in new_brand_products.index:
    old_products = old_brand_products.get(brand, set())
    new_products = new_brand_products.get(brand, set())
    added_products = new_products - old_products
    new_products_per_brand[brand] = len(added_products)

new_products_df = pd.DataFrame(list(new_products_per_brand.items()), columns=['Brand', 'New Products Count'])
new_products_df = new_products_df.sort_values(by='New Products Count', ascending=False)

# New brands growth chart
fig_new_brands = px.bar(new_brands_df.head(10),
                        x='New Brands Count', y='Category', orientation='h',
                        title='Top 10 Categories with New Brands',
                        color='New Brands Count',
                        color_continuous_scale='Teal',
                        template='plotly_white')
fig_new_brands.update_layout(yaxis={'categoryorder':'total ascending'})

# New products growth chart
fig_new_products = px.bar(new_products_df.head(10),
                          x='New Products Count', y='Brand', orientation='h',
                          title='Top 10 Brands with New Products',
                          color='New Products Count',
                          color_continuous_scale='Sunset',
                          template='plotly_white')
fig_new_products.update_layout(yaxis={'categoryorder':'total ascending'})


# Rating vs Reviews Heatmap (Scatter)
fig_rating_reviews = px.scatter(df_new, x='Stars', y='Reviews', size='Price', color='Category',
                                title='Product Reviews vs Ratings by Category', hover_name='Product Name', size_max=15)

# Top 10 Most Popular Products by Reviews
fig_top_products = px.bar(df_new.sort_values('Reviews', ascending=False).head(10),
                          x='Product Name', y='Reviews', color='Brand', title='Top 10 Most Reviewed Products')

fig_violin_price = px.violin(
    df_new, x="Category", y="Price", box=True, points="all", color="Category", title="Price Distribution Across Categories"
)
fig_violin_price.update_layout(
    xaxis_title="Category", yaxis_title="Price ($)", xaxis_tickangle=45, showlegend=False
)

fig_violin_stars = px.violin(
    df_new, x="Category", y="Weighted_Rating", box=True, points="all", color="Category", title="Weighted Rating Distribution Across Categories"
)
fig_violin_stars.update_layout(
    xaxis_title="Category", yaxis_title="Weighted Rating", xaxis_tickangle=45, showlegend=False
)

# Layout
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
                html.H1("Ulta Beauty", style={"font-size": "4rem", "font-weight": "bold", "margin": "0"}),
                html.H2("Data Analysis", style={"font-size": "2rem", "font-weight": "normal", "margin-top": "0.3rem"}),
            ]
        ),

        # Pretty Project Summary
        html.Div(
            style={
                "backgroundColor": "#FFF9F4",
                "padding": "30px 60px",
                "textAlign": "center",
                "fontFamily": "Georgia, serif",
                "color": "#444",
                "fontSize": "1.5rem",
                "lineHeight": "2",
                "maxWidth": "80%",
                "margin": "20px auto"
            },
            children=[
                html.P([
                    "This project analyzes ",
                    html.Span("Ulta Beauty", style={"fontWeight": "bold"}),
                    " product reviews to uncover insights through ",
                    html.Span("sentiment analysis,", style={"fontStyle": "italic"}),
                    " ",
                    html.Span("bias detection,", style={"fontStyle": "italic"}),
                    " pricing strategy shifts, and ",
                    html.Span("explainable machine learning.", style={"fontStyle": "italic"}),
                    " By combining ",
                    html.Span("NLP,", style={"fontStyle": "italic"}),
                    " visual analytics, and predictive modeling, we aim to support more transparent product evaluation and smarter retail decisions."
                ])
            ]
        ),

        # Visualization section
        html.Div([
            html.H3(
                "Insights Dashboard", 
            style={
                "textAlign": "center", 
                "marginTop": "40px", 
                "fontSize": "2.5rem",
                "fontWeight": "bold",
                "color": "#333",
                "font-family": "Georgia, serif"
    }
),

            html.Div([
                html.Div([
                    html.P("By scraping the real time data from ULTA website," \
                    "We compared with the dataset from 8 months ago, the first thing " \
                    "we want to know about is how many new brands and products have " \
                    "been put on the market. "),
                    dcc.Graph(figure=fig_new_brands)],
                    style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),
                html.Div([dcc.Graph(figure=fig_new_products)],)
                    
            ]),

            html.Div([
                html.Div([
                    html.P("The second thing we want to know is the current reviews " \
                    "and ratings for each category and most popular products. "),
 
                    dcc.Graph(figure=fig_rating_reviews)],
                    style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}), 
                html.Div([                  
                    html.P("As we can see, Concealer has a huge amounts of reviews compare" \
                    " to other categories. The rating for concealer is also above average." \
                    " Look at the most popular products, Tarte draw our attention. Tarte has " \
                    "the most reviews with product name 'Shape Tape Concealer', and this brand " \
                    "showed up in our previous analysis. "),

                    dcc.Graph(figure=fig_top_products),
                    html.P("Business is smart, they know to increase the quantaty of products" \
                    "when there's a good trend. Base on that, if we could dive into the data and " \
                    "take a close look at each related relationship between each feature and " \
                    "analyze them, that will be a huge help to the development of business. ")],
                    style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"}),

            ])
        ], style={"padding": "0 40px"}),

        html.Div([
            html.P("How can we inspect these features and predict future trends?" \
            " Let's have a product performance analysis:" \
            " below are the price and weighted rating distribution for each category"),
            dcc.Graph(figure=fig_violin_price),
            dcc.Graph(figure=fig_violin_stars)
        ], 
        style={"backgroundColor": "white", "padding": "25px", "borderRadius": "12px", "marginBottom": "20px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)", "fontSize": "1.2rem", "color": "#333"})
    ]
)

