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
    st.session_state.show_input_form = True

# Define input fields and preserve them using session_state
if 'show_input_form' not in st.session_state:
    st.session_state.show_input_form = False

if st.session_state.show_input_form:
    # Collect user input using Streamlit widgets and store the values in session_state
    st.header("Enter Your Details")

    if 'age' not in st.session_state:
        st.session_state.age = 0
    if 'weight' not in st.session_state:
        st.session_state.weight = 0.0
    if 'ap_hi' not in st.session_state:
        st.session_state.ap_hi = 0
    if 'ap_lo' not in st.session_state:
        st.session_state.ap_lo = 0
    if 'cholesterol' not in st.session_state:
        st.session_state.cholesterol = 1
    if 'gluc' not in st.session_state:
        st.session_state.gluc = 1

    # User input fields
    st.session_state.age = st.number_input("Age", min_value=0, value=st.session_state.age)
    st.session_state.weight = st.number_input("Weight (in kg)", min_value=0.0, step=0.1, value=st.session_state.weight)
    st.session_state.ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", min_value=0, value=st.session_state.ap_hi)
    st.session_state.ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=0, value=st.session_state.ap_lo)
    st.session_state.cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], index=st.session_state.cholesterol - 1)
    st.session_state.gluc = st.selectbox("Glucose Level", options=[1, 2, 3], index=st.session_state.gluc - 1)

    # Prediction button
    if st.button("Predict"):
        # Prepare the input data for prediction
        input_data = pd.DataFrame([[st.session_state.age, st.session_state.weight, st.session_state.ap_hi, 
                                    st.session_state.ap_lo, st.session_state.cholesterol, st.session_state.gluc]],
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