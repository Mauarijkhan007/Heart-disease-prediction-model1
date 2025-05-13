import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("Heart Disease Prediction (Mock Demo)")
st.markdown("This is a simplified preview of how the predictions will be displayed.")

# Input fields (optional, just to simulate user input)
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    submitted = st.form_submit_button("Predict")

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
