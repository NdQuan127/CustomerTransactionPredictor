import streamlit as st
import pandas as pd
from joblib import load

# Load the model and preprocessor
model = load('linear_regression_model.joblib')
preprocessor = load('preprocessor.joblib')

# Function to make predictions and recommend products
def recommend_product(current_age, per_capita_income, yearly_income, total_debt, fico_score, num_credit_cards, gender, state):
    # Create DataFrame from input data
    input_data = pd.DataFrame({
        'Current Age': [current_age],
        'Per Capita Income - Zipcode': [per_capita_income],
        'Yearly Income - Person': [yearly_income],
        'Total Debt': [total_debt],
        'FICO Score': [fico_score],
        'Num Credit Cards': [num_credit_cards],
        'Gender': [gender],
        'State': [state]
    })

    # Preprocess the input data
    input_data_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_processed)[0]

    # Recommend a product based on the predicted value
    if prediction < 100:
        product = {
            "name": "Basic - $10",
            "image": "bronze.png",
            "description": "This is an affordable product perfect for budget-conscious customers."
        }
    elif 100 <= prediction < 500:
        product = {
            "name": "Standard - $100",
            "image": "silver.png",
            "description": "This product offers a great balance of quality and price."
        }
    else:
        product = {
            "name": "Premium - $250",
            "image": "gold.png",
            "description": "This is a premium product for customers looking for the best quality."
        }

    return product, prediction

# Streamlit app layout
st.title("Average Transaction Value Prediction and Product Recommendation")

st.header("Enter Customer Details")
current_age = st.number_input("Current Age", min_value=0, max_value=120, value=25)
per_capita_income = st.number_input("Per Capita Income - Zipcode", min_value=0.0, value=30000.0)
yearly_income = st.number_input("Yearly Income - Person", min_value=0.0, value=50000.0)
total_debt = st.number_input("Total Debt", min_value=0.0, value=10000.0)
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
num_credit_cards = st.number_input("Num Credit Cards", min_value=0, max_value=10, value=2)
gender = st.selectbox("Gender", options=["Male", "Female"])
state = st.text_input("State", value="CA")

if st.button("Predict"):
    product, prediction = recommend_product(current_age, per_capita_income, yearly_income, total_debt, fico_score, num_credit_cards, gender, state)
    st.success(f"Predicted Average Transaction Value: ${prediction:.2f}")
    st.image(product["image"], caption=product["name"], use_column_width=True)
    st.info(product["description"])
