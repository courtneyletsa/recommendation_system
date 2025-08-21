# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model + preprocessing
model = joblib.load("addtocart_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")  # optional

st.title("🛒 Add-to-Cart Prediction App")

st.write("Predict whether a viewed product will be added to cart.")

# --- Input fields (example, adjust to your features) ---
categoryid = st.selectbox("Category ID", [1,2,3,4,5])  
price = st.number_input("Price", min_value=0.0, max_value=10000.0, step=1.0)
views_count = st.number_input("Number of views", min_value=0, step=1)
time_on_page = st.slider("Time on page (seconds)", 0, 600, 30)

# Create dataframe for model
input_data = pd.DataFrame({
    "categoryid": [categoryid],
    "price": [price],
    "views_count": [views_count],
    "time_on_page": [time_on_page]
})

# --- Preprocess if needed ---
if preprocessor:
    input_data = preprocessor.transform(input_data)

# --- Prediction ---
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# --- Output ---
if prediction == 1:
    st.success(f"✅ This product is likely to be added to cart (probability {probability:.2f})")
else:
    st.error(f"❌ This product is unlikely to be added to cart (probability {probability:.2f})")
