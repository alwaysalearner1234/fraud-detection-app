import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection Web App")

@st.cache_resource
def train_model():
    # Small demo dataset with 2 rows (fraud and legit)
    data = {
        "V1": [-1.359807, 1.191857],
        "V2": [-0.072781, 0.266151],
        "V3": [2.536346, 0.166480],
        "V4": [1.378155, 0.448154],
        "V5": [-0.338321, 0.060018],
        "V6": [0.462388, -0.082361],
        "V7": [-0.114805, -0.078803],
        "V8": [1.312227, 0.085102],
        "V9": [-0.615602, -0.255425],
        "V10": [-0.044074, -0.166974],
        "Amount": [149.62, 2.69],
        "Class": [0, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

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
