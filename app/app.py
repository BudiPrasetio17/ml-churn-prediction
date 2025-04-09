import sys
import os
if not os.path.exists("logs"):
    os.makedirs("logs")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
from src.predict import predict_from_input

st.set_page_config(page_title="Customer Churn Predictor")

st.title("🔮 Customer Churn Prediction App")
st.write("Masukkan data pelanggan atau upload CSV untuk prediksi churn.")

# ===============================
# 🔹 Form Prediksi Satu-per-Satu
# ===============================

st.header("📋 Prediksi Pelanggan Individu")

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
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    submitted = st.form_submit_button("Prediksi Churn")

def encode_input():
    mapping = {
        "Yes": 1, "No": 0,
        "Female": 0, "Male": 1,
        "DSL": 0, "Fiber optic": 1,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
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

if submitted:
    input_data = encode_input()
    result = predict_from_input(input_data)

    st.success(f"✅ Prediksi Churn: {'Ya' if result['churn_prediction'] else 'Tidak'}")
    st.info(f"📊 Probabilitas Churn: {result['churn_probability'] * 100:.2f}%")

# =================================
# 🔸 Prediksi Massal via CSV Upload
# =================================

st.header("📂 Prediksi Massal dari File CSV")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

def auto_encode(df):
    mapping = {
        "Yes": 1, "No": 0,
        "Female": 0, "Male": 1,
        "DSL": 0, "Fiber optic": 1,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(mapping).fillna(df[col])

    return df

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)

        # Hapus baris header duplikat jika ada
        if df_input.iloc[0].equals(df_input.columns.to_series()):
            df_input = df_input.drop(index=0).reset_index(drop=True)

        # Auto-encode kolom kategorikal
        df_input = auto_encode(df_input)

        expected_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]

        if all(col in df_input.columns for col in expected_cols):
            model = joblib.load("models/churn_model.pkl")
            predictions = model.predict(df_input)
            probs = model.predict_proba(df_input)[:, 1]

            df_input["Churn_Prediction"] = predictions

                    # Simpan ke file log
            log_file = "logs/prediction_log.csv"
            df_log = df_input.copy()
            df_log["timestamp"] = pd.Timestamp.now()

            if os.path.exists(log_file):
                df_log.to_csv(log_file, mode='a', index=False, header=False)
            else:
                df_log.to_csv(log_file, index=False)
            
            st.info(f"📁 Hasil prediksi berhasil disimpan ke `{log_file}`")


            df_input["Churn_Probability"] = probs.round(2)

            st.success("✅ Prediksi berhasil!")
            st.dataframe(df_input)

                    # Pie Chart Churn
            st.subheader("📊 Distribusi Prediksi Churn")
            churn_counts = df_input["Churn_Prediction"].value_counts()
            churn_labels = ["Tidak Churn", "Churn"]
            churn_values = [churn_counts.get(0, 0), churn_counts.get(1, 0)]

            st.pyplot(
                pd.Series(churn_values, index=churn_labels).plot.pie(
                    autopct='%1.1f%%',
                    ylabel='',
                    title='Proporsi Churn vs Tidak Churn',
                    figsize=(4, 4)
                ).figure
            )

                    # Bar Chart Probabilitas
            st.subheader("📈 Distribusi Probabilitas Churn")
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
            df_input["ProbGroup"] = pd.cut(df_input["Churn_Probability"], bins=bins, labels=labels)

            st.bar_chart(df_input["ProbGroup"].value_counts().sort_index())



            csv_download = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Hasil", csv_download, "prediksi_churn.csv", "text/csv")

        else:
            st.error("❌ Kolom tidak cocok. Pastikan CSV punya semua kolom yang diperlukan.")

    except Exception as e:
        st.error(f"❌ Terjadi error saat membaca file: {e}")

                # Divider
        st.markdown("---")
        st.subheader("🕘 Riwayat Prediksi Sebelumnya")

        log_file = "logs/prediction_log.csv"

        if os.path.exists(log_file):
            df_log = pd.read_csv(log_file)

            # Opsi sort by waktu terbaru
            df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
            df_log = df_log.sort_values(by="timestamp", ascending=False)

            st.dataframe(df_log, use_container_width=True)

            st.markdown("---")
            
            st.subheader("🔎 Filter Riwayat Prediksi")

            if os.path.exists(log_file):
                df_log = pd.read_csv(log_file)
                df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
                df_log["date"] = df_log["timestamp"].dt.date

                # 📅 Pilih rentang tanggal
                min_date = df_log["date"].min()
                max_date = df_log["date"].max()

                start_date = st.date_input("Dari Tanggal", min_value=min_date, max_value=max_date, value=min_date)
                end_date = st.date_input("Sampai Tanggal", min_value=min_date, max_value=max_date, value=max_date)

                # 🔀 Filter Churn / Tidak Churn
                filter_churn = st.selectbox("Filter Churn", options=["Semua", "Churn Saja", "Tidak Churn Saja"])

                # 🧼 Terapkan filter
                filtered_log = df_log[(df_log["date"] >= start_date) & (df_log["date"] <= end_date)]

                if filter_churn == "Churn Saja":
                    filtered_log = filtered_log[filtered_log["Churn_Prediction"] == 1]
                elif filter_churn == "Tidak Churn Saja":
                    filtered_log = filtered_log[filtered_log["Churn_Prediction"] == 0]

                st.write(f"📊 Menampilkan {len(filtered_log)} baris hasil prediksi:")
                st.dataframe(filtered_log, use_container_width=True)
            else:
                st.info("Belum ada log untuk difilter.")


            st.markdown("---")
            st.subheader("📈 Tren Churn Harian")

            if os.path.exists(log_file):
                df_log = pd.read_csv(log_file)
                df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])

                # Grouping by tanggal
                df_log["date"] = df_log["timestamp"].dt.date
                churn_trend = df_log.groupby(["date", "Churn_Prediction"]).size().unstack(fill_value=0)

                st.line_chart(churn_trend.rename(columns={0: "Tidak Churn", 1: "Churn"}))


            # Tombol download log
            csv_log = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Log", csv_log, "riwayat_prediksi.csv", "text/csv")
        else:
            st.info("Belum ada riwayat prediksi yang tersimpan.")

            
