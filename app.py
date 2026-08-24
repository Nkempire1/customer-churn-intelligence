
import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# LOAD THE TRAINED MODEL
# ------------------------------------------------------------

model = joblib.load("customer_churn_model.pkl")


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("Customer Churn Intelligence")
st.write(
    "Predict customer churn risk and prioritize retention efforts."
)


# ------------------------------------------------------------
# CUSTOMER INFORMATION
# ------------------------------------------------------------

st.header("Customer Information")


col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col2:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if st.button("Predict Churn Risk"):

    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })


    probability = model.predict_proba(
        customer
    )[0, 1]


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if probability >= 0.80:
        risk = "CRITICAL RISK"

    elif probability >= 0.60:
        risk = "HIGH RISK"

    elif probability >= 0.40:
        risk = "MEDIUM RISK"

    else:
        risk = "LOW RISK"


    # --------------------------------------------------------
    # REVENUE EXPOSURE
    # --------------------------------------------------------

    revenue_at_risk = (
        monthly_charges * probability
    )


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.header("Prediction Results")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk
        )


    with result_col3:

        st.metric(
            "Estimated Monthly Revenue at Risk",
            f"${revenue_at_risk:,.2f}"
        )


    if probability >= 0.60:

        st.error(
            "This customer should be considered for retention intervention."
        )

    else:

        st.success(
            "This customer is currently below the retention intervention threshold."
        )
