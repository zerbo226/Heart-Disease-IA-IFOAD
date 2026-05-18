import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# Load the trained model and dataset
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_pred_model.pkl")

@st.cache_data
def load_data():
    # Use the finalized dataset for visualizations
    return pd.read_csv("heart_disease_data_finalized.csv")

model = load_model()
df = load_data()

# Streamlit UI
st.title("Heart Disease Prediction & Dashboard")

# Create tabs to fulfill both the prediction and dashboard requirements
tab1, tab2 = st.tabs(["🩺 Prediction Tool", "📊 Dashboard (Key Findings)"])

with tab1:
    st.header("Prediction Tool")
    st.write("Enter the patient's details to predict heart disease severity.")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalch = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Slope of Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels (0-3) Colored by Fluoroscopy", min_value=0, max_value=3, value=1)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert categorical inputs to numerical values
    sex_num = 1 if sex == "Male" else 0
    cp_num = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)
    fbs_num = 1 if fbs == "True" else 0
    restecg_num = ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    exang_num = 1 if exang == "Yes" else 0
    slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
    thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame(np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num, restecg_num, thalch, exang_num, oldpeak, slope_num, ca, thal_num]]),
                              columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"])

    # Ensure all data is numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')

    # Handle any missing values (e.g., fill with 0 or mean)
    input_data = input_data.fillna(0)

    # Predict
    if st.button("Predict Heart Disease"):
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            result = "No heart disease (Healthy heart)"
            st.success(f"Prediction: {result}")
        else:
            result = f"Heart disease detected (Severity Level: {prediction[0]})"
            st.error(f"Prediction: {result}")

with tab2:
    st.header("Interactive Data Dashboard")
    st.write("Explore the key findings and trends from the heart disease dataset.")
    
    # We create two columns for better layout of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age Distribution Histogram
        fig_age = px.histogram(df, x="age", color="num", barmode="group",
                               title="Age Distribution by Heart Disease Severity",
                               labels={"num": "Severity (0=No Disease)", "age": "Age"})
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Chest Pain Type Bar Chart
        fig_cp = px.histogram(df, x="cp", color="num", barmode="group",
                              title="Chest Pain Type vs Heart Disease",
                              labels={"cp": "Chest Pain Type", "num": "Severity"})
        st.plotly_chart(fig_cp, use_container_width=True)

    with col2:
        # Cholesterol vs Age Scatter Plot
        fig_chol = px.scatter(df, x="age", y="chol", color="num", 
                              title="Cholesterol Levels by Age and Heart Disease",
                              labels={"chol": "Cholesterol (mg/dl)", "age": "Age", "num": "Severity"})
        st.plotly_chart(fig_chol, use_container_width=True)
        
        # Max Heart Rate vs Age Scatter Plot
        fig_thalach = px.scatter(df, x="age", y="thalch", color="num",
                                 title="Max Heart Rate by Age and Heart Disease",
                                 labels={"thalch": "Max Heart Rate", "age": "Age", "num": "Severity"})
        st.plotly_chart(fig_thalach, use_container_width=True)