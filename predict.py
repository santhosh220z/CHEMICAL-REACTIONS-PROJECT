# app.py

from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)

# Load model
MODEL_PATH = 'ann_model_reactions.keras'
model = load_model(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

# Load and fit scaler on training data (ideally load saved scaler.pkl)
df = pd.read_csv('reactions.csv')
features = ['temperature', 'pH', 'concentration']
X = df[features].values
scaler = StandardScaler()
scaler.fit(X)

# Home page — serve HTML
@app.route('/')
def home():
    return render_template('index.html')

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        temp = float(request.form['temperature'])
        ph = float(request.form['ph'])
        conc = float(request.form['concentration'])

        # Scale input
        input_scaled = scaler.transform([[temp, ph, conc]])

        # Predict
        prediction = model.predict(input_scaled)[0][0]
        success = 1 if prediction > 0.5 else 0

        result_text = f"Success Probability: {prediction:.4f} → Prediction: {'Success' if success==1 else 'Failure'}"

        return jsonify({'result': result_text})

    except Exception as e:
        print("Error:", e)
        return jsonify({'result': 'Error in prediction. Please check input values.'})

if __name__ == '__main__':
    app.run(debug=True)
