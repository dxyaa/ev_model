from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the trained model
model_path = 'C:/Users/Diya/projects/ev_charging_model_rf.joblib'
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame(data)

    # Fill missing values with the mean of the column
    input_df.fillna(input_df.mean(), inplace=True)

    # Features for prediction
    X_input = input_df[['remaining_battery', 'drain_rate', 'remaining_range', 'estimated_time_left', 'time_to_station', 'distance_to_station']]

    # Predict priorities
    predicted_priorities = model.predict(X_input)

    # Add predicted priorities to input DataFrame
    input_df['predicted_priority'] = predicted_priorities

    # Sort DataFrame by predicted priority
    priority_list = input_df.sort_values(by='predicted_priority', ascending=False)

    # Get car_id with highest priority
    max_priority_car_id = priority_list.iloc[0]['car_id']

    return jsonify({'car_id': int(max_priority_car_id)})

if __name__ == '__main__':
    app.run(debug=True)
