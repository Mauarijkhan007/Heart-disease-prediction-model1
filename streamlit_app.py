import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load models
chi2_dnn = load_model("chi2_dnn_model.h5")
chi2_ann = load_model("chi2_ann_model.h5")
dnn = load_model("dnn_model.h5")
ann = load_model("ann_model.h5")

chi2_selected_features = ['sex', 'cp', 'trestbps', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction App")

with st.form("input_form"):
    st.subheader("Enter Patient Data")
    age = st.number_input("Age", min_value=1, max_value=120, value=21)
    sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=180)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=170)
    exang = st.radio("Exercise Induced Angina", options=[0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, step=0.1, value=0.0)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed", 7: "Reversible"}[x])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Prepare input
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    df = pd.DataFrame([data])
    chi2_df = df[chi2_selected_features]
    full_df = df[all_features]
    chi2_scaled = StandardScaler().fit_transform(MinMaxScaler().fit_transform(chi2_df))
    full_scaled = StandardScaler().fit_transform(full_df)

    # Prediction function
    def get_label(prob): return "Unhealthy" if prob > 0.5 else "Healthy"

    # Get predictions
    preds = {
        "χ²-DNN": chi2_dnn.predict(chi2_scaled, verbose=0)[0][0],
        "χ²-ANN": chi2_ann.predict(chi2_scaled, verbose=0)[0][0],
        "DNN": dnn.predict(full_scaled, verbose=0)[0][0],
        "ANN": ann.predict(full_scaled, verbose=0)[0][0]
    }

    # Display model outputs
    st.subheader("Individual Model Outputs")
    for model_name, prob in preds.items():
        st.write(f"**{model_name}:** {get_label(prob)}")

    # Ensemble verdict
    avg_prob = np.mean(list(preds.values()))
    st.subheader("Final Verdict")
    if avg_prob < 0.55:
        st.success("Result: Healthy")
    elif avg_prob <= 0.60:
        st.warning("Result: Uncertain – Recommend clinical testing")
    else:
        st.error("Result: Unhealthy")
