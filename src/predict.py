# src/predict.py

import pandas as pd
import joblib

def predict_from_input(input_dict: dict):
    # Load model
    model = joblib.load("models/churn_model.pkl")

    # Convert input jadi DataFrame
    df = pd.DataFrame([input_dict])

    # Prediksi
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # Probabilitas churn

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability, 2)
    }
