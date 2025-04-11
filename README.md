# -student-streamlitApp.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(page_title="Timelytics", page_icon=":pencil:", layout="wide")

# Title and description
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques.")

st.caption(
    "Timelytics is an ensemble model that utilizes XGBoost, Random Forests, and SVM to forecast Order to Delivery (OTD) times. "
    "It helps businesses identify bottlenecks, reduce lead times, and improve delivery accuracy."
)

# Load the trained ensemble model
@st.cache_resource
def load_model():
    modelfile = "E:/Sonu11/Intel AI/Week6/Week 6 - Dependencies/Dependencies/voting_model.pkl"
    return pickle.load(open(modelfile, "rb"))

voting_model = load_model()

# Wait time prediction function
def waitime_predictor(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance,
):
    features = np.array([[
        purchase_dow,
        purchase_month,
        year,
        product_size_cm3,
        product_weight_g,
        geolocation_state_customer,
        geolocation_state_seller,
        distance
    ]])
    prediction = voting_model.predict(features)
    return round(prediction[0])

# Sidebar inputs
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")

    purchase_dow = st.number_input("Purchased Day of the Week (0=Mon)", min_value=0, max_value=6, step=1, value=3)
    purchase_month = st.number_input("Purchased Month", min_value=1, max_value=12, step=1, value=1)
    year = st.number_input("Purchased Year", value=2018)
    product_size_cm3 = st.number_input("Product Size (cm³)", value=9328)
    product_weight_g = st.number_input("Product Weight (g)", value=1800)
    geolocation_state_customer = st.number_input("Customer State (Encoded)", value=10)
    geolocation_state_seller = st.number_input("Seller State (Encoded)", value=20)
    distance = st.number_input("Distance (km)", value=475.35)
    submit = st.button("Predict Wait Time!")

# Main panel
with st.container():
    st.header("📦 Predicted Delivery Time")

    if submit:
        with st.spinner("Predicting..."):
            prediction = waitime_predictor(
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance,
            )
        st.success(f"🕒 Estimated Wait Time: {prediction} days")

    # Sample dataset section
    st.header("📊 Sample Dataset (Preview)")
    data = {
        "Purchased Day of the Week": [0, 3, 1],
        "Purchased Month": [6, 3, 1],
        "Purchased Year": [2018, 2017, 2018],
        "Product Size in cm^3": [37206.0, 63714, 54816],
        "Product Weight in grams": [16250.0, 7249, 9600],
        "Geolocation State Customer": [25, 25, 25],
        "Geolocation State Seller": [20, 7, 20],
        "Distance": [247.94, 250.35, 4.915],
    }
    df = pd.DataFrame(data)
    st.dataframe(df)
