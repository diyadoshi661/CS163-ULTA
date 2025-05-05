import dash
from dash import html, dcc, Input, Output
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import numpy as np
dash.register_page(__name__, path='/predictions', name='Predictions')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(BASE_DIR,'..', 'data')

# Load and prepare data
csv_path = os.path.join(csv_dir, 'cleaned_makeup_products.csv')
df_old = pd.read_csv(csv_path)

# Data cleaning
df = df_old.drop([
    'product_link_id', 'product_link', 'item_id', 'description', 
    'pros', 'cons', 'best_uses', 'describe_yourself', 
    'native_sampling_review_count', 'faceoff_negative', 'faceoff_positive'
], axis=1)

df = df.dropna(subset=['average_rating'])
df['brand'] = df['brand'].fillna('Unknown')
df['category'] = df['category'].fillna('Unknown')

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(0)
# Remove string versions of NaN
df['product_name'] = df['product_name'].astype(str)
df['product_name'] = df['product_name'].str.replace(r'^nan\s+', '', regex=True).replace('nan', '')
df['product_name'] = df['product_name'].replace('', 'Unknown')


label_encoders = {}
for col in ['brand', 'category', 'product_name']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('average_rating', axis=1)
y = df['average_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Prepare DataFrame for visualization
results_df = pd.DataFrame({
    'Actual Rating': y_test,
    'Predicted Rating': y_pred
})

# Scatter plot: Actual vs Predicted
fig1 = px.scatter(
    results_df,
    x='Actual Rating',
    y='Predicted Rating',
    title='Actual vs Predicted Product Ratings',
    labels={'Actual Rating': 'Actual Rating', 'Predicted Rating': 'Predicted Rating'},
    opacity=0.6
)

# Interactive prediction setup
# We will let user adjust 'price' between min and max
price_min = int(df['price'].min())
price_max = int(df['price'].max())

# Create sample data for different prices (other features are fixed at average values)
def predict_for_prices(price_values):
    input_df = pd.DataFrame(np.tile(X.mean().values, (len(price_values), 1)), columns=X.columns)
    input_df['price'] = price_values
    preds = model.predict(input_df)
    return preds


layout = html.Div(
    style={"backgroundColor": "#FFF9F4", "padding": "30px", "font-family": "Georgia, serif"},
    children=[
        # Title
        html.Div(
    style={
        "backgroundColor": "#BFA2DB",   # ✨ Soft purple strip
        "padding": "20px",
        "textAlign": "center",
        "borderRadius": "10px",
        "marginBottom": "30px",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
    },
    children=html.H1(
        "Predictions",
        style={
            "fontSize": "3rem",
            "fontWeight": "bold",
            "color": "white",
            "margin": "0",
            "font-family": "Georgia, serif"
        }
    )
),


        # Project Summary Card
        html.Div([
            html.H2("Goal", style={"color": "#4B0082", "marginBottom": "10px"}),
            html.P("With the question in mind: Can we predict the rating according to price? " \
            "what is the best price we should set up for basic products? " \
            "In this project, we developed a Random Forest Regressor "
                   "to predict average product ratings based on the average product price. "
                    "The goal was to explore whether structured features like price,"
                     " could effectively forecast user ratings. ",
                   style={"color": "#333", "fontSize": "1.2rem"})
        ], style={"backgroundColor": "white", "padding": "20px", "borderRadius": "12px", "width": "600px", "margin": "20px auto", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}),

        # Data Sources Card
       
        html.H3("Product Rating Prediction Visualization"),
    
        html.P("This scatter plot (Actual vs Predicted Ratings) visualize the accuracy of prediction"),
        dcc.Graph(figure=fig1),

        html.Hr(),

        html.H4("Predict Rating Based on Price"),

        dcc.Slider(
            id='price-slider',
            min=price_min,
            max=price_max,
            step=1,
            value=(price_min + price_max) // 2,  # start at mid price
            marks={price_min: str(price_min), price_max: str(price_max)},
            tooltip={"placement": "bottom", "always_visible": True}
         ),

        dcc.Graph(id='interactive-prediction-graph'),
        html.P("As we can see from the graph, predicted ratings increase slightly as price increases" \
        ", but the change is marginal. The rating does not increase linearly with price-plateaus and " \
        "jumps suggest price sensitivity zones. ",style={"color": "#333", "fontSize": "1.6rem"}),
        html.Div([
        html.H4(" Business Strategy Recommendations", style={"marginBottom": "20px"}),

        html.P("1. Price Optimization\n"
           "• Sweet Spot Zone: Price points between $30–$50 slightly increase perceived product rating.\n"
           "• High-End Saturation: Pricing above $50 doesn't provide much improvement in predicted rating. (" \
           "That way we can focus on differentiators like luxury packaging or exclusive branding instead."),

    html.P("2. Value Perception Campaigns\n"
           "• Since rating does slightly increase with price, position slightly more expensive items as premium-quality, we can also use marketing method to enhance perceived value.\n"
           "• Consider A/B testing campaigns around $29 vs $45 product tiers."),

    html.P("3.  Targeted Promotions\n"
           "• For products priced below $10, consider bundling or upselling — these are associated with the lowest predicted ratings, potentially perceived as “cheap” or lower quality."),

    html.P("4. Product Line Segmentation\n"
           "• Entry-level (< $15): Emphasize affordability, everyday use.\n"
           "• Core tier ($30–$50): Your main offering, aligned with highest predicted ratings.\n"
           "• Prestige tier ($50+): Justify the price with influencer partnerships, packaging, or exclusivity — " \
           "that's why ratings likely won’t rise much, so non-rating perks must be visible."),


    
], style={
    "backgroundColor": "white",
    "padding": "25px",
    "borderRadius": "12px",
    "marginBottom": "20px",
    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
    "fontSize": "1.1rem",
    "color": "#333",
    "lineHeight": "1.8"
})

        
])

# Callback for interactive prediction
@dash.callback(
    Output('interactive-prediction-graph', 'figure'),
    Input('price-slider', 'value')
)
def update_prediction(selected_price):
    price_values = np.linspace(price_min, price_max, 100)
    preds = predict_for_prices(price_values)

    fig2 = px.line(
        x=price_values,
        y=preds,
        labels={'x': 'Price', 'y': 'Predicted Rating'},
        title=f"Predicted Rating vs Price (Selected Price: ${selected_price})"
    )
    fig2.add_vline(x=selected_price, line_dash="dash", line_color="red")
    return fig2
