import dash
from dash import html

dash.register_page(__name__, path='/methods', name='Analytical Methods')

layout = html.Div([
    html.H2("Analytical Methods"),
    html.P("This section outlines the key data science and machine learning techniques used in the project."),
    html.Ul([
        html.Li("Data Preprocessing: Cleaning reviews, handling missing data, label encoding categorical features."),
        html.Li("Sentiment Analysis: Using VADER/TextBlob or DistilBERT to quantify sentiment of review text."),
        html.Li("Keyword Extraction: Using TF-IDF or KeyBERT to find common terms per brand or category."),
        html.Li("XGBoost Regression: Modeling the star rating using features like price, brand, category, and review count."),
        html.Li("SHAP: Visualizing feature importance for model explainability."),
    ]),
    html.P("See the Findings page for insights and visuals derived from these methods."),
])

