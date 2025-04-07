# app/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import predict_from_input

st.set_page_config(page_title="Customer Churn Predictor")

st.title("🔮 Customer Churn Prediction")
st.write("Masukkan detail pelanggan untuk memprediksi kemungkinan churn.")

# Form input
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (bulan)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiline = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    onlinesec = st.selectbox("Online Security", ["No", "Yes"])
    onlinebackup = st.selectbox("Online Backup", ["No", "Yes"])
    deviceprotect = st.selectbox("Device Protection", ["No", "Yes"])
    techsupport = st.selectbox("Tech Support", ["No", "Yes"])
    streamtv = st.selectbox("Streaming TV", ["No", "Yes"])
    streammovie = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    submitted = st.form_submit_button("Prediksi Churn")

# Encoding sesuai model
def encode_input():
    mapping = {
        "Yes": 1, "No": 0,
        "Female": 0, "Male": 1,
        "No internet service": 0,
        "No phone service": 0,
        "DSL": 0, "Fiber optic": 1,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }

    return {
        "gender": mapping[gender],
        "SeniorCitizen": mapping[senior],
        "Partner": mapping[partner],
        "Dependents": mapping[dependents],
        "tenure": tenure,
        "PhoneService": mapping[phone],
        "MultipleLines": mapping[multiline],
        "InternetService": mapping[internet],
        "OnlineSecurity": mapping[onlinesec],
        "OnlineBackup": mapping[onlinebackup],
        "DeviceProtection": mapping[deviceprotect],
        "TechSupport": mapping[techsupport],
        "StreamingTV": mapping[streamtv],
        "StreamingMovies": mapping[streammovie],
        "Contract": mapping[contract],
        "PaperlessBilling": mapping[paperless],
        "PaymentMethod": mapping[payment],
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

# Proses prediksi
if submitted:
    input_data = encode_input()
    result = predict_from_input(input_data)

    st.success(f"✅ Prediksi Churn: {'Ya' if result['churn_prediction'] else 'Tidak'}")
    st.info(f"📊 Probabilitas Churn: {result['churn_probability'] * 100:.2f}%")
