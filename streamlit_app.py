import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction")

# Input form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest pain type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=250)
    chol = st.number_input("Serum cholesterol (mg/dl)", min_value=100, max_value=600)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=250)
    exang = st.selectbox("Exercise induced angina", options=[0, 1])
    oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, step=0.1)
    slope = st.selectbox("Slope of the ST segment", options=[0, 1, 2])
    ca = st.selectbox("Number of major vessels (0–3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x])

    submitted = st.form_submit_button("Predict")

# Output section with simulated predictions
if submitted:
    st.subheader("Individual Model Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("χ²-DNN", "Healthy")
        st.metric("DNN", "Healthy")
    with col2:
        st.metric("χ²-ANN", "Healthy")
        st.metric("ANN", "Uncertain")

    st.markdown("---")
    st.subheader("Final Verdict")
    st.success("Result: Healthy")
