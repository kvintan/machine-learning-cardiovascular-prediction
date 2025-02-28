from flask import Flask, render_template, request
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = load('model.joblib')
scaler = load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html', current_page='home')  # Your HTML form

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/about-us')
def aboutus():
    return render_template('about-us.html', current_page='about-us')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from user form (adjust based on form fields)
    age = int(request.form['age'])
    weight = float(request.form['weight'])
    ap_hi = int(request.form['ap_hi'])  # Systolic blood pressure
    ap_lo = int(request.form['ap_lo'])  # Diastolic blood pressure
    cholesterol = int(request.form['cholesterol'])  # Cholesterol level
    gluc = int(request.form['gluc'])  # Glucose level

    # Create a DataFrame for prediction (make sure these are the same columns as the training data)
    input_data = pd.DataFrame([[age, weight, ap_hi, ap_lo, cholesterol, gluc]],
                              columns=['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc'])

    # Apply the same scaling that was used during training (only for the columns that remain)
    input_data = scaler.transform(input_data)  # Scale the necessary features

    # Predict using the model with scaled data
    prediction = model.predict(input_data)
    prediction = prediction[0]
  
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
