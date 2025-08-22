# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("addtocart_model.pkl")
# preprocessor = joblib.load("preprocessor.pkl")  # if you saved preprocessing separately

st.title("🛒 Add-to-Cart Prediction App")

st.write("Predict whether a viewed product with the following properties will be added to cart")

# --- Input fields ---
hour = st.number_input("Hour of the day (0–23)", min_value=0, max_value=23, step=1)
dayofweek = st.selectbox("Day of the week", [0, 1, 2, 3, 4, 5, 6])  # 0=Monday ... 6=Sunday
time_since_listing = st.number_input("Time Since Listing (days)", min_value=0, max_value=10000, step=1)
visitor_item_views = st.number_input("Visitor - Item Interaction Count", min_value=0, max_value=10000, step=1)
item_popularity = st.number_input("Item Popularity (total views)", min_value=0, max_value=100000, step=1)

# Create dataframe for model input
input_data = pd.DataFrame({
    "hour": [hour],
    "dayofweek": [dayofweek],
    "time_since_listing": [time_since_listing],
    "visitor_item_views": [visitor_item_views],
    "item_popularity": [item_popularity]
})

# --- Prediction button ---
if st.button("🔮 Predict"):
    # If you had a preprocessor, apply it here
    # if preprocessor:
    #     input_data = preprocessor.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ This product is likely to be added to cart")
    else:
        st.error("❌ This product is unlikely to be added to cart")
