# src/data_prep.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop customerID karena tidak berguna
    df.drop("customerID", axis=1, inplace=True)

    # Ganti TotalCharges yang kosong jadi NaN, lalu convert ke float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Isi NaN dengan median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Encode target: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Label Encoding semua kolom object (kecuali target)
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
