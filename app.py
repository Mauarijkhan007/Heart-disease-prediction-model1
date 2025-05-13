import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load models with compile=False for compatibility
chi2_dnn = load_model("chi2_dnn_model.h5", compile=False)
chi2_ann = load_model("chi2_ann_model.h5", compile=False)
dnn = load_model("dnn_model.h5", compile=False)
ann = load_model("ann_model.h5", compile=False)

# Features
chi2_selected_features = ['sex', 'cp', 'trestbps', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
all_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction App")

with st.form("input_form"):
    st.subheader("Enter Patient Data")
    age = st.number_input("Age", 1, 120, 21)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 180)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 170)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed", 7: "Reversible"}[x])
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    df = pd.DataFrame([input_data])
    chi2_df = df[chi2_selected_features]
    full_df = df[all_features]

    chi2_scaled = StandardScaler().fit_transform(MinMaxScaler().fit_transform(chi2_df))
    full_scaled = StandardScaler().fit_transform(full_df)

    def get_label(prob): return "Unhealthy" if prob > 0.5 else "Healthy"

    preds = {
        "χ²-DNN": get_label(chi2_dnn.predict(chi2_scaled, verbose=0)[0][0]),
        "χ²-ANN": get_label(chi2_ann.predict(chi2_scaled, verbose=0)[0][0]),
        "DNN": get_label(dnn.predict(full_scaled, verbose=0)[0][0]),
        "ANN": get_label(ann.predict(full_scaled, verbose=0)[0][0])
    }

    st.subheader("Individual Model Outputs")
    for model, label in preds.items():
        st.write(f"**{model}:** {label}")

    avg_prob = np.mean([
        chi2_dnn.predict(chi2_scaled, verbose=0)[0][0],
        chi2_ann.predict(chi2_scaled, verbose=0)[0][0],
        dnn.predict(full_scaled, verbose=0)[0][0],
        ann.predict(full_scaled, verbose=0)[0][0]
    ])

    st.subheader("Final Ensemble Verdict")
    if avg_prob < 0.55:
        st.success("Result: Healthy")
    elif avg_prob <= 0.60:
        st.warning("Result: Uncertain – Recommend clinical testing")
    else:
        st.error("Result: Unhealthy")
