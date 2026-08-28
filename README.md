# DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction

Absolutely. Here is a **clean, professional, copy-paste-ready `README.md`** specifically for your **DeepCSAT Prediction** project. I’ve kept the terminology consistent with your notebook and the features/model you actually built. 

Copy **everything inside the code block** into your GitHub `README.md`.

````markdown
# ⭐ DeepCSAT Prediction Engine

## Deep Learning-Based Customer Satisfaction (CSAT) Score Prediction

A Deep Learning project that predicts **Customer Satisfaction (CSAT) scores from 1 to 5** using customer service interaction data, feature engineering, sentiment analysis, and a custom **Attention-Residual Artificial Neural Network (ANN)**.

The project also includes an interactive **Gradio web application** that allows users to enter customer and interaction details and receive a predicted CSAT score in real time.

---

## 📌 Project Overview

Customer Satisfaction (CSAT) is an important metric used to evaluate the quality of customer service and understand customer experience.

This project develops an end-to-end machine learning and deep learning pipeline to predict CSAT scores using:

- Customer interaction information
- Product information
- Customer remarks
- Sentiment analysis
- Response time
- Order-to-issue delay
- Time-based features
- Customer and agent-related categorical features

The final system provides an interactive interface where users can enter customer information and receive a predicted CSAT score between **1 and 5**.

---

## 🎯 Project Objective

The main objective of this project is to build a Deep Learning model capable of predicting customer satisfaction based on customer service interaction data.

### Target Variable

```text
CSAT Score
````

The model predicts a continuous score which is then limited to the valid CSAT range:

```text
1.0 – 5.0
```

---

## 📊 Dataset

After data cleaning, the dataset contains:

```text
Records: 85,907
```

The dataset includes customer service interaction, product, agent, timing, sentiment, and CSAT information.

### Main Original Features

* `Unique id`
* `channel_name`
* `category`
* `Sub-category`
* `Customer Remarks`
* `Order_id`
* `order_date_time`
* `Issue_reported at`
* `issue_responded`
* `Survey_response_Date`
* `Customer_City`
* `Product_category`
* `Item_price`
* `Agent_name`
* `Supervisor`
* `Manager`
* `Tenure Bucket`
* `Agent Shift`
* `CSAT Score`

---

# 🧹 Data Cleaning

The data preparation stage includes validation and cleaning of the customer service dataset.

The cleaning process checks for:

* Duplicate records
* Missing values
* Highly incomplete columns
* Missing CSAT scores
* Data consistency

After cleaning, the final dataset contains:

```text
85,907 records
```

No records were removed because of missing CSAT scores in the final cleaning stage.

---

# 🧩 Feature Engineering

Additional features were created to improve the predictive capability of the Deep Learning model.

## ⏱️ Time-Based Features

The following features were engineered from the interaction timestamps:

```text
response_time_min
order_to_issue_delay_hrs
issue_hour
issue_dayofweek
is_weekend
```

### `response_time_min`

Represents the customer service response/handling time in minutes.

### `order_to_issue_delay_hrs`

Represents the time difference between the order and the reported issue.

### `issue_hour`

Extracts the hour at which the issue was reported.

### `issue_dayofweek`

Represents the day of the week when the issue was reported.

### `is_weekend`

Indicates whether the issue occurred during the weekend.

---

# 💬 Sentiment Analysis

Customer remarks were analyzed using **VADER Sentiment Analysis**.

Three sentiment-related features were generated:

```text
sentiment_compound
sentiment_neg
sentiment_pos
```

These features capture the overall sentiment and the negative/positive components of customer feedback.

---

# 📝 Text Feature

The length of the customer's remark was also calculated:

```text
remark_length
```

This feature represents the number of characters in the customer remark.

---

# 🔢 Model Input Features

The final model uses **17 input features**.

## Numerical Features

```text
Item_price
response_time_min
order_to_issue_delay_hrs
issue_hour
issue_dayofweek
is_weekend
remark_length
sentiment_compound
sentiment_neg
sentiment_pos
```

## Categorical Features

```text
channel_name
category
Sub-category
Customer_City
Product_category
Tenure Bucket
Agent Shift
```

The preprocessing pipeline transforms these features before they are passed to the neural network.

---

# ⚙️ Data Preprocessing

The dataset is divided into training and testing sets.

```text
Training records: 68,725
Testing records: 17,182
```

Categorical and numerical features are processed using the preprocessing pipeline before being passed to the Deep Learning model.

The resulting feature matrices contain:

```text
Training feature matrix: (68,725, 1,724)

Testing feature matrix:  (17,182, 1,724)
```

The preprocessing pipeline is fitted using the training data and then applied to the testing data.

---

# 🧠 Deep Learning Model Architecture

The project uses a custom **Attention-Residual Artificial Neural Network**.

The architecture contains the following major components:

```text
Input Features
      ↓
Feature Attention Gate
      ↓
Dense Block 1
      ↓
Dense Block 2
      ↓
Residual / Skip Connection
      ↓
Dense Bottleneck
      ↓
Output Layer
      ↓
Predicted CSAT Score
```

---

## 1️⃣ Feature Attention Gate

The model first creates an attention vector using a sigmoid-activated Dense layer.

The attention mechanism learns which input features are more important for the prediction.

```text
Input
  ↓
Dense(input_dim)
  ↓
Sigmoid
  ↓
Feature Attention
  ↓
Multiply with Input
```

---

## 2️⃣ Dense Block 1

The first Dense block contains:

```text
Dense(128)
Swish Activation
Batch Normalization
Dropout(0.3)
```

---

## 3️⃣ Dense Block 2

The second Dense block contains:

```text
Dense(128)
Swish Activation
Batch Normalization
Dropout(0.3)
```

A residual connection is then applied:

```text
x1 + x2
```

This creates a skip connection that helps information flow through the network.

---

## 4️⃣ Bottleneck Layer

The residual output is passed through:

```text
Dense(64)
Swish Activation
Batch Normalization
```

---

## 5️⃣ Output Layer

The final layer contains:

```text
Dense(1)
Linear Activation
```

The output represents the predicted CSAT score.

The final prediction is clipped to:

```text
1.0 – 5.0
```

---

# 🏋️ Model Training

The neural network is trained using the following configuration:

| Parameter         | Value                    |
| ----------------- | ------------------------ |
| Optimizer         | AdamW                    |
| Learning Rate     | 0.001                    |
| Weight Decay      | 0.0001                   |
| Loss Function     | Mean Squared Error (MSE) |
| Batch Size        | 256                      |
| Maximum Epochs    | 50                       |
| Activation        | Swish                    |
| Dropout           | 0.3                      |
| Output Activation | Linear                   |

---

# ⏳ Training Optimization

Two callbacks are used during training.

## Early Stopping

Training stops when validation loss stops improving.

```text
patience = 8
```

The best model weights are restored after training.

## Learning Rate Reduction

The learning rate is reduced when validation loss stops improving.

```text
factor = 0.5
patience = 3
minimum learning rate = 1e-6
```

---

# 📈 Model Evaluation

The trained model is evaluated using three metrics:

## Mean Absolute Error (MAE)

Measures the average absolute difference between the actual and predicted CSAT scores.

```text
MAE = average absolute prediction error
```

## Root Mean Squared Error (RMSE)

Measures prediction error while giving greater importance to larger errors.

```text
RMSE = √MSE
```

## R² Score

Measures how well the model explains the variation in the target variable.

The notebook reports:

```text
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
```

---

# 📊 Training Curves

The project also visualizes:

### Loss Curve

Shows:

* Training MSE
* Validation MSE

### MAE Curve

Shows:

* Training MAE
* Validation MAE

These plots help evaluate model convergence and identify potential overfitting.

---

# 🌐 Interactive Gradio Web Application

The project includes an interactive web interface built with **Gradio**.

Users can enter customer and interaction information and receive a predicted CSAT score.

## Input Features

The application accepts:

* Channel Name
* Category
* Sub-category
* Customer City
* Product Category
* Tenure Bucket
* Agent Shift
* Item Price
* Handling Time
* Customer Remark

---

# 🔮 Prediction Example

### Example Input

```text
Channel Name:
Email

Category:
App/website

Product Category:
Affiliates

Tenure Bucket:
0-30

Agent Shift:
Afternoon

Item Price:
150

Handling Time:
120 seconds

Customer Remark:
The agent was very helpful and resolved my issue quickly.
```

### Example Output

```text
⭐ Predicted CSAT Score: 4.28 / 5.0
```

The prediction value is generated by the trained Deep Learning model and may vary depending on the model state and input values.

---

# 🔄 End-to-End Pipeline

The complete project follows this workflow:

```text
Raw Customer Service Data
          ↓
Data Cleaning
          ↓
Data Validation
          ↓
Feature Engineering
          ↓
VADER Sentiment Analysis
          ↓
Time-Based Feature Engineering
          ↓
Train/Test Split
          ↓
Numerical & Categorical Preprocessing
          ↓
Attention-Residual ANN
          ↓
Model Training
          ↓
Model Evaluation
          ↓
CSAT Prediction
          ↓
Gradio Web Application
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn

### Deep Learning

* TensorFlow
* Keras

### Natural Language Processing

* NLTK
* VADER Sentiment Analysis

### Visualization

* Matplotlib

### Web Application

* Gradio

### Development Environment

* Google Colab
* Jupyter Notebook

---

# 📁 Project Structure

```text
DeepCSAT-Prediction/
│
├── README.md
│
├── DeepLearnCSAT-prediction.ipynb
│
├── requirements.txt
│
├── .gitignore
│
├── models/
│   └── csat_model.keras
│
└── screenshots/
    ├── model_results.png
    └── gradio_app.png
```

> The `models/` and `screenshots/` folders are optional if those files are not included in the repository.

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepCSAT-Prediction.git
```

Navigate to the project directory:

```bash
cd DeepCSAT-Prediction
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Open the Notebook

Open:

```text
DeepLearnCSAT-prediction.ipynb
```

The notebook can be executed using:

* Google Colab
* Jupyter Notebook
* JupyterLab

---

## 4. Run the Notebook Cells in Order

Run the project stages sequentially:

```text
Cell 1 → Data Loading
Cell 2 → Data Cleaning
Cell 3 → Feature Engineering & Sentiment Analysis
Cell 4 → Data Preprocessing
Cell 5 → Deep Learning Model Training
Cell 6 → Model Evaluation
Cell 7 → Gradio Web Application
```

---

# 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
scikit-learn
tensorflow
nltk
matplotlib
gradio
```

---

# 🔐 Data Privacy

The original customer dataset is **not included in this repository**.

This is intentional because customer service datasets may contain sensitive or proprietary information.

Users who want to reproduce the project should provide an appropriate dataset with the required columns.

---

# 🚀 Future Improvements

Potential future improvements include:

* Hyperparameter optimization
* Cross-validation
* Advanced NLP models
* Transformer-based sentiment analysis
* Explainable AI using SHAP
* Feature importance visualization
* Real-time prediction API
* Cloud deployment
* Model monitoring
* Customer satisfaction dashboards
* Automated model retraining
* REST API integration

---

# ⭐ Key Project Highlights

* ✅ 85,907 cleaned customer service records
* ✅ 17 model input features
* ✅ Time-based feature engineering
* ✅ VADER sentiment analysis
* ✅ Numerical and categorical preprocessing
* ✅ Attention mechanism
* ✅ Residual/skip connection
* ✅ Deep Learning ANN
* ✅ AdamW optimizer
* ✅ Early stopping
* ✅ Learning-rate scheduling
* ✅ MAE evaluation
* ✅ RMSE evaluation
* ✅ R² evaluation
* ✅ Interactive Gradio application
* ✅ End-to-end CSAT prediction pipeline

---

# 👩‍💻 Author

## Neha Lahane

Machine Learning | Deep Learning | Data Analytics

---

# 📌 Disclaimer

This project is developed for educational, analytical, and demonstration purposes.

The predicted CSAT score should be treated as a model-generated estimate and should not be considered a guaranteed representation of actual customer satisfaction.

---





