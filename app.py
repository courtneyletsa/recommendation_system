# app.py
import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("addtocart_model.pkl")
# preprocessor = joblib.load("preprocessor.pkl")  # if you saved preprocessing separately

# --- Prediction function ---
def predict(hour, dayofweek, time_since_listing, visitor_item_views, item_popularity):
    input_data = pd.DataFrame({
        "hour": [hour],
        "dayofweek": [dayofweek],
        "time_since_listing": [time_since_listing],
        "visitor_item_views": [visitor_item_views],
        "item_popularity": [item_popularity]
    })

    # If you had a preprocessor, apply it here
    # if preprocessor:
    #     input_data = preprocessor.transform(input_data)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "✅ This product is likely to be added to cart"
    else:
        return "❌ This product is unlikely to be added to cart"

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛒 Add-to-Cart Prediction App")
    gr.Markdown("Predict whether a viewed product will be added to cart")

    with gr.Row():
        hour = gr.Number(label="Hour of the day (0–23)", value=0)
        dayofweek = gr.Dropdown([0, 1, 2, 3, 4, 5, 6], label="Day of the week (0=Mon, 6=Sun)", value=0)

    with gr.Row():
        time_since_listing = gr.Number(label="Time Since Listing (days)", value=0)
        visitor_item_views = gr.Number(label="Visitor - Item Interaction Count", value=0)

    item_popularity = gr.Number(label="Item Popularity (total views)", value=0)

    predict_btn = gr.Button("🔮 Predict")
    output = gr.Textbox(label="Prediction Result")

    predict_btn.click(
        fn=predict,
        inputs=[hour, dayofweek, time_since_listing, visitor_item_views, item_popularity],
        outputs=output
    )

# Run the app
if __name__ == "__main__":
    demo.launch(share=True)

