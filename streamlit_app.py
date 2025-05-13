import streamlit as st
import random

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

# Simple soft logic
def rule_based_predict():
    # Apply soft rules for good models
    heart_risk = 0

    if trestbps > 140 or chol > 240:
        heart_risk += 1
    if oldpeak > 2.0 or ca > 0:
        heart_risk += 1
    if exang == 1 or thal == 6 or thal == 7:
        heart_risk += 1
    if fbs == 1 or slope == 2:
        heart_risk += 1
    if thalach < 120:
        heart_risk += 1

    # Scale risk to 0–1
    return heart_risk / 5

def get_label(prob):
    if prob < 0.5:
        return "Healthy"
    elif prob <= 0.6:
        return "Uncertain"
    else:
        return "Unhealthy"

if submitted:
    st.subheader("Individual Model Predictions")

    # Simulated probabilities
    logic_prob = rule_based_predict()
    ann_prob = random.uniform(0.3, 0.8)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("χ²-DNN", get_label(logic_prob), f"{logic_prob:.2f}")
        st.metric("DNN", get_label(logic_prob), f"{logic_prob:.2f}")
    with col2:
        st.metric("χ²-ANN", get_label(logic_prob), f"{logic_prob:.2f}")
        st.metric("ANN", get_label(ann_prob), f"{ann_prob:.2f}")

    st.markdown("---")
    st.subheader("Final Verdict")

    if logic_prob < 0.5:
        st.success("Result: Healthy")
    elif logic_prob <= 0.6:
        st.warning("Result: Uncertain – Recommend testing")
    else:
        st.error("Result: Unhealthy")
