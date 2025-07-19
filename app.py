import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load trained Isolation Forest model
model = joblib.load('fraud_detection_model.pkl')

st.title("💳 Financial Fraud Detection Dashboard")

uploaded_file = st.file_uploader("Upload Transaction CSV File", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Raw Data:")
    st.dataframe(data.head())

    if st.button("Predict Fraud"):

        df = data.copy()

        # Scale 'Amount' and 'Time' just like in training
        scaler_amount = StandardScaler()
        scaler_time = StandardScaler()

        df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))

        # Drop columns not used for prediction
        df = df.drop(['Amount', 'Time', 'Class'], axis=1, errors='ignore')

        # Arrange columns to match training order: scaled_time, scaled_amount, remaining features
        remaining_cols = [col for col in df.columns if col not in ['scaled_time', 'scaled_amount']]
        expected_cols = ['scaled_time', 'scaled_amount'] + remaining_cols

        # Confirm all expected columns exist
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            features = df[expected_cols]

            # Make Predictions
            y_pred = model.predict(features)
            y_pred = [1 if x == -1 else 0 for x in y_pred]  # Outliers (-1) are frauds

            data['Fraud_Prediction'] = y_pred
            st.write("Prediction Results with Fraud Labels:")
            st.dataframe(data.head())

            fraud_count = sum(y_pred)
            st.success(f"Total Fraudulent Transactions Detected: {fraud_count}")

            # Plot Fraud Distribution
            fig, ax = plt.subplots()
            sns.countplot(x=y_pred, palette='Set2')
            plt.title("Fraud (1) vs Legit (0) Transactions")
            st.pyplot(fig)

st.write("Sample Fraudulent Transactions:")
st.dataframe(data[data['Fraud_Prediction'] == 1].head())