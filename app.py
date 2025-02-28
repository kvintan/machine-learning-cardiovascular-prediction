from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import streamlit as st

# Load the trained model and scaler
model = load('model.joblib')
scaler = load('scaler.joblib')

# Inject custom CSS for styling the navbar and other elements
st.markdown("""
    <style>
        .navbar {
            background-color: #f8f9fa;
            padding: 10px;
        }
        .navbar-logo {
            width: 25vw;
            margin-top: 0.4vw;
        }
        .homepage-start {
            text-align: center;
            margin-top: 30px;
        }
        .homepage-jantung {
            width: 200px;
            margin: 20px 0;
        }
        .homepage-cardiovascular-prediction {
            font-size: 3em;
            font-weight: bold;
        }
        .homepage-check-heart {
            font-size: 1.5em;
            margin-bottom: 30px;
        }
        .homepage-button-start-check {
            padding: 10px 20px;
            font-size: 1.2em;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
        }
        .homepage-button-start-check:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)


# Homepage section (Main content for the homepage)
st.markdown("""
    <div class="homepage-start">
        <h1 class="homepage-cardiovascular-prediction">
            Cardiovascular<br />Prediction
        </h1>
        <h3 class="homepage-check-heart">
            Check Your Heart Health Risks with AI!
        </h3>
    </div>
    """, unsafe_allow_html=True)

# "Start Check" button that links to the input page
if st.button('Start Check'):
    st.write("Redirecting to the input page...")
    # You can directly call the input form function here or navigate to another page.

    # Alternatively, show the input form directly in the same page
    st.header("Enter Your Details")

    # Collect user input using Streamlit widgets
    age = st.number_input("Age", min_value=0)
    weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1)
    ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=0)
    ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=0)
    cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3])
    gluc = st.selectbox("Glucose Level", options=[1, 2, 3])

    # Prediction button
    if st.button("Predict"):
        # Prepare the input data for prediction
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
