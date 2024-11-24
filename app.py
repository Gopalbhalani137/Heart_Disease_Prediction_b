from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load

app = Flask(__name__)
CORS(app)  # This enables CORS for all origins

# Path to the saved model
MODEL_PATH = 'model/trained_model.joblib'

# Load the model using joblib
model = load(MODEL_PATH)

@app.route('/')
def home():
    return jsonify({"message": "Heart Disease Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
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
    app.run(port=3000)  # Set the port to 3000