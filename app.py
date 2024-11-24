from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://capable-youtiao-2f285b.netlify.app/","http://localhost:5173", "http://127.0.0.1:5173"]}})
# Path to the saved model - using os.path for better compatibility
MODEL_PATH = os.path.join('model', 'trained_model.joblib')

try:
    # Load the model using joblib
    model = load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    if model is None:
        return jsonify({"message": "Warning: Model not loaded", "status": "error"}), 500
    return jsonify({"message": "Heart Disease Prediction API is running.", "status": "success"})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        print("data in backend", data)
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = {
            "prediction": int(prediction[0]),
            "probability": {
                "no_heart_disease": float(prediction_proba[0][0]),
                "heart_disease": float(prediction_proba[0][1]),
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
