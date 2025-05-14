import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger information below:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 32.2)
embarked = st.selectbox("Port of Embarkation", ['S', 'C', 'Q'])

# Convert to model input format
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [0 if sex == 'male' else 1],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [ {'S': 0, 'C': 1, 'Q': 2}[embarked] ]
})

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🟢 The passenger **would survive**.")
    else:
        st.error("🔴 The passenger **would not survive**.")
