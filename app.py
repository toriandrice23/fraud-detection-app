import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Medical Claim Fraud Detection App")

uploaded_file = st.file_uploader("Upload a claim data file (CSV or Excel)", type=["csv", "xlsx"])

@st.cache_resource
def load_model():
    return joblib.load("final_logistic_model.joblib")

model = load_model()

def preprocess(df):
    if 'claim_amount' in df.columns:
        df['claim_amount'] = df['claim_amount'].fillna(0)
    return df.select_dtypes(include=[np.number])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.write(df.head())

        preprocessed = preprocess(df)
        preds = model.predict(preprocessed)
        probs = model.predict_proba(preprocessed)[:, 1]

        df_results = df.copy()
        df_results['Fraud_Prediction'] = preds
        df_results['Fraud_Probability'] = probs

        st.subheader("Prediction Results")
        st.write(df_results)

        csv_download = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", csv_download, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
