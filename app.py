
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection Web App")

# Load the trained model
model = joblib.load("fraud_model.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(data)

    if st.button("Predict Fraud"):
        preds = model.predict(data)
        data['Prediction'] = ['Fraud' if p == 1 else 'Legit' for p in preds]

        st.subheader("🔍 Results")
        st.dataframe(data[['Prediction']])

        st.success(f"✅ Detected {sum(preds)} fraudulent transactions out of {len(preds)}.")
