import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("rf_model.pkl")

st.title("🚢 Titanic Survival Prediction")

# User inputs (same features as training)
p_class = st.selectbox("Passenger Class (p_class)", [1, 2, 3])
sex = st.selectbox("Sex", ["female", "male"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sib_sp = st.number_input("Siblings / Spouses aboard (sib_sp)", min_value=0, value=0)
parch = st.number_input("Parents / Children aboard (parch)", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)

# Encoding (same logic used during training)
# female = 0, male = 1
sex = 0 if sex == "female" else 1

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[p_class, sex, age, sib_sp, parch, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
