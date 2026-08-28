
# ============================================================
# DEEPCSAT - CUSTOMER SATISFACTION PREDICTION APPLICATION
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ============================================================
# 1. DOWNLOAD VADER SENTIMENT DICTIONARY
# ============================================================

nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()


# ============================================================
# 2. LOAD TRAINED MODEL AND PREPROCESSOR
# ============================================================

MODEL_PATH = "csat_model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# 3. FEATURES USED DURING TRAINING
# ============================================================

numerical_features = [
    "Item_price",
    "response_time_min",
    "order_to_issue_delay_hrs",
    "issue_hour",
    "issue_dayofweek",
    "is_weekend",
    "remark_length",
    "sentiment_compound",
    "sentiment_neg",
    "sentiment_pos"
]

categorical_features = [
    "channel_name",
    "category",
    "Sub-category",
    "Customer_City",
    "Product_category",
    "Tenure Bucket",
    "Agent Shift"
]

feature_columns = numerical_features + categorical_features


# ============================================================
# 4. CSAT PREDICTION FUNCTION
# ============================================================

def predict_csat(
    channel_name,
    category,
    sub_category,
    customer_city,
    product_category,
    tenure_bucket,
    agent_shift,
    item_price,
    handling_time,
    customer_remark
):

    # --------------------------------------------------------
    # SENTIMENT ANALYSIS
    # --------------------------------------------------------

    customer_remark = str(customer_remark)

    sentiment_scores = sia.polarity_scores(customer_remark)

    sentiment_compound = sentiment_scores["compound"]
    sentiment_neg = sentiment_scores["neg"]
    sentiment_pos = sentiment_scores["pos"]

    remark_length = len(customer_remark)


    # --------------------------------------------------------
    # DEFAULT OPERATIONAL FEATURES
    # --------------------------------------------------------

    # These are default values because the deployed UI does
    # not ask the user for timestamps.
    response_time_min = 15.0
    order_to_issue_delay_hrs = 24.0
    issue_hour = 12
    issue_dayofweek = 1
    is_weekend = 0


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    sample = pd.DataFrame([{

        # Numerical features
        "Item_price": float(item_price),

        "response_time_min": response_time_min,

        "order_to_issue_delay_hrs":
            order_to_issue_delay_hrs,

        "issue_hour": issue_hour,

        "issue_dayofweek": issue_dayofweek,

        "is_weekend": is_weekend,

        "remark_length": remark_length,

        "sentiment_compound":
            sentiment_compound,

        "sentiment_neg":
            sentiment_neg,

        "sentiment_pos":
            sentiment_pos,

        # Categorical features
        "channel_name": channel_name,

        "category": category,

        "Sub-category": sub_category,

        "Customer_City": customer_city,

        "Product_category": product_category,

        "Tenure Bucket": tenure_bucket,

        "Agent Shift": agent_shift
    }])


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    transformed = preprocessor.transform(
        sample[feature_columns]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        transformed,
        verbose=0
    )[0][0]


    # Keep CSAT within valid 1-5 range
    score = float(
        np.clip(prediction, 1.0, 5.0)
    )

    score = round(score, 2)


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return f"⭐ Predicted CSAT Score: {score} / 5.0"


# ============================================================
# 5. GRADIO WEB INTERFACE
# ============================================================

demo = gr.Interface(

    fn=predict_csat,

    inputs=[

        gr.Dropdown(
            choices=[
                "Outcall",
                "Inbound",
                "In-app Chat",
                "Email"
            ],
            label="Channel Name",
            value="Email"
        ),

        gr.Dropdown(
            choices=[
                "App/website",
                "Product Queries",
                "Returns",
                "Order Tracking",
                "Refunds"
            ],
            label="Category",
            value="App/website"
        ),

        gr.Textbox(
            label="Sub-category",
            value="Account updation"
        ),

        gr.Textbox(
            label="Customer City",
            value="ABOHAR"
        ),

        gr.Textbox(
            label="Product Category",
            value="Affiliates"
        ),

        gr.Dropdown(
            choices=[
                "0-30",
                "31-60",
                "61-90",
                "90+",
                "On Job Training"
            ],
            label="Tenure Bucket",
            value="0-30"
        ),

        gr.Dropdown(
            choices=[
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ],
            label="Agent Shift",
            value="Afternoon"
        ),

        gr.Number(
            label="Item Price",
            value=150
        ),

        gr.Number(
            label="Handling Time (seconds)",
            value=120
        ),

        gr.Textbox(
            label="Customer Remark",
            value=(
                "The agent was very helpful "
                "and resolved my issue quickly."
            ),
            lines=4
        )
    ],

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="🔮 DeepCSAT Prediction Engine",

    description=(
        "Customer Satisfaction Score Prediction "
        "using a Deep Learning Residual Attention "
        "Artificial Neural Network."
    )
)


# ============================================================
# 6. LAUNCH
# ============================================================

demo.launch()
