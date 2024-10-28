import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('loan_predictor_model.pkl', 'rb') as file:
    model_info = pickle.load(file)
    model = model_info['model']
    scaler = model_info['scaler']
    feature_names = model_info['feature_names']

# Title of the app
st.title("Loan Status Prediction App")

# Input features with more understandable names
no_of_dependents = st.number_input("Number of Dependents", min_value=0, value=1, help="Total number of dependents (e.g., children, spouse)")
income_annum = st.number_input("Annual Income (in INR)", min_value=0, value=3000000, help="Total annual income of the applicant")
loan_amount = st.number_input("Requested Loan Amount (in INR)", min_value=0, value=5000, help="Amount of loan requested by the applicant")
loan_term = st.number_input("Loan Term (in Years)", min_value=0, value=2, help="Duration of the loan in years")
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, help="Credit score ranging from 300 to 900")
residential_assets_value = st.number_input("Value of Residential Assets (in INR)", min_value=0, value=5000000, help="Total value of residential properties owned")
commercial_assets_value = st.number_input("Value of Commercial Assets (in INR)", min_value=0, value=75000, help="Total value of commercial properties owned")
luxury_assets_value = st.number_input("Value of Luxury Assets (in INR)", min_value=0, value=0, help="Total value of luxury items owned (e.g., cars, jewelry)")
bank_asset_value = st.number_input("Value of Bank Assets (in INR)", min_value=0, value=30000, help="Total value of savings or investments in banks")

# Create a feature array
user_input = np.array([[no_of_dependents, income_annum, loan_amount, loan_term,
                        cibil_score, residential_assets_value,
                        commercial_assets_value, luxury_assets_value,
                        bank_asset_value]])

# Scale the features
user_input_scaled = scaler.transform(user_input)

# Predict loan status
if st.button("Predict Loan Status"):
    predicted = model.predict(user_input_scaled)
    predicted_status = "Approved" if predicted[0] == 1 else "Rejected"
    
    # Change background color based on prediction
    if predicted_status == "Approved":
        st.markdown(
            """
            <style>
            .reportview-container {
                background: linear-gradient(to right, #c6f8c6, #a1e1a1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .reportview-container {
                background: linear-gradient(to right, #f8d6d6, #e1a1a1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.success(f'Predicted Loan Status: {predicted_status}')
