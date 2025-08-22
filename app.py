# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model + preprocessing
model = joblib.load("addtocart_model.pkl")
# preprocessor = joblib.load("preprocessor.pkl")  # optional

st.title("🛒 Add-to-Cart Prediction App")

st.write("Predict whether a viewed product with the following properties will be added to cart.")

# --- Input fields (example, adjust to your features) ---
# categoryid = st.selectbox("Category ID", [1,2,3,4,5])  
hour = st.number_input("Hours", min_value=0.0, max_value=10000.0, step=1.0)
dayofweek = st.selectbox("Day of the week", [0,1,2,3,4,5,6])
time_since_listing = st.number_input("Time Since Listing", min_value=0.0, max_value=10000.0, step=1.0)
visitor_item_views = st.number_input("Visitor - Item Interaction Count", min_value=0.0, max_value=10000.0, step=1.0)
item_popularity = st.number_input("Item popularity", min_value=0.0, max_value=10000.0, step=1.0)
# time_on_page = st.slider("Time on page (seconds)", 0, 600, 30)

# Create dataframe for model
input_data = pd.DataFrame({
#     "categoryid": [categoryid],
    "hour": [hour],
    "dayofweek": [dayofweek],
    "time_since_listing": [time_since_listing],
    "visitor_item_views": [visitor_item_views],
    "item_popularity": [item_popularity]
})

# --- Preprocess if needed ---
# if preprocessor:
#     input_data = preprocessor.transform(input_data)

# --- Prediction ---
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# --- Output ---
if prediction == 1:
    st.success(f"✅ This product is likely to be added to cart (probability {probability:.2f})")
else:
    st.error(f"❌ This product is unlikely to be added to cart (probability {probability:.2f})")
