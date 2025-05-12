import dash
from dash import html


# Register the page
dash.register_page(__name__, path='/proposal', name='Proposal')

layout = html.Div(
    style={"backgroundColor": "#FFF9F4", "padding": "30px"},
    children=[
        # Header
        html.Div(
            style={
                "backgroundColor": "#bfa2db",
                "padding": "20px",
                "textAlign": "center",
                "borderRadius": "10px",
                "marginBottom": "30px"
            },
            children=[
                html.H1("Project Proposal", style={"color": "white", "fontFamily": "Georgia, serif"})
            ]
        ),

        # Project Summary
        html.Div([
            html.H2("Project Summary", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.P("""
                This project aims to build a comprehensive Ulta Beauty product analytics platform by combining sentiment and bias detection, pricing strategy analysis, and explainable machine learning modeling.
We scrape, clean, and process thousands of product listings and customer reviews to uncover how consumer opinions, price changes, and brand dynamics affect overall product performance.
Our goal is to enable more transparent product evaluation, strategic retail decisions, and better consumer trust.

            """, style={"fontSize": "18px", "color": "#333", "marginTop": "10px"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Sources
        html.Div([
    html.H2("Data Sources", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
    html.P("Ulta Beauty Website (Scraped Data): Products and reviews were scraped directly from Ulta Beauty’s Face Makeup Section. "
           "This dataset includes product information such as product names, prices, ratings, review counts, and brand details, "
           "enabling analysis of makeup products available on the platform.",
           style={"marginBottom": "20px", "fontSize": "18px", "color": "#333"}),

    html.P("Kaggle Dataset: Makeup Insights - Customer Reviews: Provides a large-scale structured dataset from 8 months ago containing "
           "review-level data across makeup products. Key fields include review ID, product link ID, review text headline, customer nickname, "
           "location, product page ID, UPC codes, verified buyer status, and helpfulness scores, facilitating detailed review sentiment and behavior analysis.",
           style={"marginBottom": "20px", "fontSize": "18px", "color": "#333"}),

    html.P("Kaggle Dataset: Ulta Skincare Reviews: Offers over 4,000 detailed reviews for Dermalogica cleansing exfoliators on Ulta.com. "
           "The dataset captures review titles, full review text, verification status, review dates, locations, upvotes, downvotes, associated products, brands, "
           "and scrape dates, providing a focused view for skincare-specific customer feedback analysis.",
           style={"marginBottom": "0px", "fontSize": "18px", "color": "#333"}),
], style={
    "backgroundColor": "white",
    "padding": "20px",
    "borderRadius": "10px",
    "marginBottom": "20px",
    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
}),

html.Hr(style={"borderTop": "2px solid #bfa2db"}),


        # Expected Major Findings
        html.Div([
    html.H2("Expected Major Findings", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
    html.Ul([
        html.Li("Sentiment and Bias Patterns: Reviews often show emotional bias, especially among unverified buyers. Over 40% of reviews had high subjectivity scores (>0.6), suggesting inflated sentiment that could distort real product ratings.", style={"marginBottom": "8px"}),
        html.Li("Explainable ML for Star Ratings: Using XGBoost, we achieved RMSE = 0.2305 and MAE = 0.1575 in predicting product star ratings. Features like review volume and price had the biggest impact, revealed via SHAP analysis.", style={"marginBottom": "8px"}),
        html.Li("Cluster Analysis of Reviews: Using KMeans and Radar plots, we detected 3 behavior clusters — (1) Short and neutral filler reviews, (2) Detailed authentic reviews, and (3) Overly positive emotional reviews (potential exaggerations).", style={"marginBottom": "8px"})
    ], style={"fontSize": "18px", "color": "#333"})
], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Preprocessing Steps
        html.Div([
            html.H2("Preprocessing Steps", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Data Collection and Cleaning", style={"marginBottom": "8px"}),
                html.Li("Scaling and Normalization", style={"marginBottom": "8px"}),
                html.Li("Feature Engineering and Selection")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Analysis and Algorithms
        html.Div([
            html.H2("Data Analysis and Algorithms", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Sentiment and Subjectivity Scoring", style={"marginBottom": "8px"}),
                html.Li("Cluster Analysis for Bias Detection", style={"marginBottom": "8px"}),
                html.Li("XGBoost Modeling and SHAP Analysis"),
                html.Li("Random Forest Regression Model for Product Rating Prediction")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Hr(style={"borderTop": "2px solid #bfa2db"}),

        # Data Visualization Plan
        html.Div([
            html.H2("Data Visualization Plan", style={"color": "#4b0082", "fontFamily": "Georgia, serif"}),
            html.Ul([
                html.Li("Violin plots for price and rating distributions", style={"marginBottom": "8px"}),
                html.Li("Radar charts for review behavior clustering", style={"marginBottom": "8px"}),
                html.Li("SHAP summary plots for explainability"),
                html.Li(" a scatter plot comparing actual vs predicted ratings, demonstrating the model’s predictive ability."),
                html.Li("Enabled users to adjust the price using a slider and observe how the predicted rating changes, based on the model's learned patterns.")
            ], style={"fontSize": "18px", "color": "#333"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "marginBottom": "40px"}),
    ]
)
