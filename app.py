from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import streamlit as st

# Load the trained model and scaler (make sure the files are in the correct path)
model = load('model.joblib')
scaler = load('scaler.joblib')

# Streamlit interface
st.title("Cardiovascular Disease Prediction")

# Collect user input using Streamlit widgets
age = st.number_input("Age", min_value=0)
weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1)
ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=0)
ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=0)
cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3])
gluc = st.selectbox("Glucose Level", options=[1, 2, 3])

# Prediction button
if st.button("Predict"):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[age, weight, ap_hi, ap_lo, cholesterol, gluc]],
                              columns=['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc'])

    # Apply the same scaling that was used during training
    input_data = scaler.transform(input_data)

    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.write("You have a **high risk** of cardiovascular disease.")
    else:
        st.write("You have a **low risk** of cardiovascular disease.")

