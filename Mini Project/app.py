from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and pipeline properly
with open("disease_model.pkl", "rb") as f:
    saved = pickle.load(f)
    model = saved["model"]
    label_encoder = saved["label_encoder"]

with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Define expected states (for dropdown)
expected_columns = [
    'state_ut_Andhra Pradesh', 'state_ut_Arunachal Pradesh', 'state_ut_Assam', 'state_ut_Bihar',
    'state_ut_Chandigarh', 'state_ut_Chhattisgarh', 'state_ut_Dadra and Nagar Haveli',
    'state_ut_Daman and Diu', 'state_ut_Delhi', 'state_ut_Goa', 'state_ut_Gujarat',
    'state_ut_Haryana', 'state_ut_Himachal Pradesh', 'state_ut_Jammu and Kashmir',
    'state_ut_Jharkhand', 'state_ut_Karnataka', 'state_ut_Kerala', 'state_ut_Lakshadweep',
    'state_ut_Madhya Pradesh', 'state_ut_Maharashtra', 'state_ut_Manipur', 'state_ut_Meghalaya',
    'state_ut_Mizoram', 'state_ut_Nagaland', 'state_ut_Odisha', 'state_ut_Puducherry',
    'state_ut_Punjab', 'state_ut_Rajasthan', 'state_ut_Sikkim', 'state_ut_Tamil Nadu',
    'state_ut_Telangana', 'state_ut_Tripura', 'state_ut_Uttar Pradesh', 'state_ut_Uttarakhand',
    'state_ut_West Bengal'
]
states = [col.replace('state_ut_', '') for col in expected_columns]

@app.route('/')
def home():
    return render_template('index.html', states=states)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        inputs = {
            'week_of_outbreak': int(request.form['week']),
            'district': str(request.form['district']),
            'Cases': float(request.form['cases']),
            'Deaths': float(request.form['deaths']),
            'day': int(request.form['day']),
            'mon': int(request.form['mon']),
            'year': int(request.form['year']),
            'Latitude': float(request.form['lat']),
            'Longitude': float(request.form['lon']),
            'preci': float(request.form['preci']),
            'LAI': float(request.form['lai']),
            'Temp': float(request.form['temp']),
            'state_ut': str(request.form['state'])
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([inputs])

        # Preprocess
        processed = preprocessor.transform(input_df)

        # Predict
        pred_proba = model.predict_proba(processed)
        pred_idx = np.argmax(pred_proba, axis=1)
        pred_label = label_encoder.inverse_transform(pred_idx)[0]
        pred_prob = np.max(pred_proba)

        # Format result
        result_text = f"Probability of {pred_label} occurring is {pred_prob:.2f}"

        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
