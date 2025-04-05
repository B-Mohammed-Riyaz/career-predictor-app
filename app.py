import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("career_prediction_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🚀 Career Predictor App")
st.write("Predict what you might become when you grow up based on your inputs!")

# Get the columns except the target one
input_columns = list(label_encoders.keys())
input_columns.remove("What would you like to become when you grow up")

# Collect user inputs
user_input = {}
for col in input_columns:
    le = label_encoders[col]
    options = le.classes_
    choice = st.selectbox(f"{col}", options)
    user_input[col] = le.transform([choice])[0]

# Predict
if st.button("Predict Career"):
    input_df = pd.DataFrame([user_input])
    prediction_encoded = model.predict(input_df)[0]
    prediction_label = label_encoders["What would you like to become when you grow up"].inverse_transform([prediction_encoded])[0]
    st.success(f"🎯 You might become: **{prediction_label}**")
