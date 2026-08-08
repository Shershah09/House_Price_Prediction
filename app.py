# ==========================================================
# Imports
# ==========================================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# Load Pipeline
# ==========================================================

pipeline = joblib.load("house_price_pipeline.pkl")


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# ==========================================================
# Header
# ==========================================================

st.title("🏠 House Price Prediction")

st.write(
    "Predict Boston house prices using Machine Learning "
    "with a Gradient Boosting Regressor."
)

st.divider()


# ==========================================================
# Model Information
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Gradient Boosting")

with col2:
    st.metric("R² Score", "0.881")

with col3:
    st.metric("RMSE", "2.998")


st.divider()


# ==========================================================
# 50 : 50 Layout
# ==========================================================


left_col, right_col = st.columns([1, 1], gap="large")

# ==========================================================
# LEFT SIDE - House Details
# ==========================================================

with left_col:

    st.subheader("🏡 House Details")
    st.caption("Enter the house features below.")
    st.divider()

    user_input = {}

    user_input["CRIM"] = st.slider(
        "Crime Rate",
        min_value=0.006,
        max_value=88.98,
        value=0.25
    )

    user_input["ZN"] = st.slider(
        "Residential Land (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

    user_input["INDUS"] = st.slider(
        "Industrial Area",
        min_value=0.46,
        max_value=27.74,
        value=9.69
    )

    user_input["CHAS"] = st.selectbox(
        "Charles River",
        options=[0, 1],
        help="1 = Near River, 0 = Not Near River"
    )

    user_input["NOX"] = st.slider(
        "Nitric Oxide",
        min_value=0.38,
        max_value=0.87,
        value=0.54
    )

    user_input["RM"] = st.slider(
        "Average Rooms",
        min_value=3.5,
        max_value=9.0,
        value=6.20
    )

    user_input["AGE"] = st.slider(
        "House Age",
        min_value=2.0,
        max_value=100.0,
        value=77.5
    )

    user_input["DIS"] = st.slider(
        "Distance",
        min_value=1.1,
        max_value=12.2,
        value=3.2
    )

    user_input["RAD"] = st.slider(
        "Highway Access",
        min_value=1,
        max_value=24,
        value=5
    )

    user_input["TAX"] = st.slider(
        "Property Tax",
        min_value=187,
        max_value=711,
        value=330
    )

    user_input["PTRATIO"] = st.slider(
        "Student-Teacher Ratio",
        min_value=12.6,
        max_value=22.0,
        value=18.9
    )

    user_input["B"] = st.slider(
        "Black Population Index",
        min_value=0.32,
        max_value=396.90,
        value=391.0
    )

    user_input["LSTAT"] = st.slider(
        "Lower Status Population (%)",
        min_value=1.7,
        max_value=38.0,
        value=11.3
    )

    input_df = pd.DataFrame([user_input])

    predict_button = st.button(
        "🔮 Predict House Price",
        use_container_width=True
    )


# ==========================================================
# RIGHT SIDE - Results
# ==========================================================

with right_col:

    st.subheader("💰 Estimated House Price")

    if predict_button:

        prediction = pipeline.predict(input_df)

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="Predicted House Price",
            value=f"${prediction[0]:.2f}K"
        )

    else:

        st.info(
            "Enter the house details and click "
            "**Predict House Price**."
        )

    st.divider()

    # ======================================================
    # Feature Importance
    # ======================================================

    st.subheader("📊 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": [
            "LSTAT",
            "RM",
            "DIS",
            "PTRATIO",
            "CRIM",
            "NOX",
            "AGE",
            "TAX",
            "B",
            "RAD",
            "INDUS",
            "CHAS",
            "ZN"
        ],
        "Importance": [
            0.47,
            0.32,
            0.09,
            0.03,
            0.025,
            0.021,
            0.015,
            0.013,
            0.006,
            0.002,
            0.001,
            0.0008,
            0.00008
        ]
    })

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.divider()

    # ======================================================
    # About Project
    # ======================================================

    st.subheader("ℹ️ About Project")

    st.markdown("""
### Dataset
- Boston Housing Dataset

### Machine Learning Model
- Gradient Boosting Regressor

### Performance
- **R² Score:** 0.881
- **RMSE:** 2.998

### Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Train-Test Split
6. Pipeline & ColumnTransformer
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Streamlit Deployment
""")