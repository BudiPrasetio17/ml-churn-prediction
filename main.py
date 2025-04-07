# main.py

from src.train_model import train
# from src.predict import predict_from_input

if __name__ == "__main__":
    train()

# Contoh input
# sample = {
#     'gender': 1,
#     'SeniorCitizen': 0,
#     'Partner': 1,
#     'Dependents': 0,
#     'tenure': 5,
#     'PhoneService': 1,
#     'MultipleLines': 0,
#     'InternetService': 1,
#     'OnlineSecurity': 0,
#     'OnlineBackup': 1,
#     'DeviceProtection': 0,
#     'TechSupport': 0,
#     'StreamingTV': 1,
#     'StreamingMovies': 1,
#     'Contract': 0,
#     'PaperlessBilling': 1,
#     'PaymentMethod': 2,
#     'MonthlyCharges': 70.35,
#     'TotalCharges': 350.2
# }

# result = predict_from_input(sample)
# print("Churn Prediction:", result["churn_prediction"])
# print("Churn Probability:", result["churn_probability"])