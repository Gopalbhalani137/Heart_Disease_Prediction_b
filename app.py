from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app) 
MODEL_PATH = 'model/trained_model.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return jsonify({"message": "Heart Disease Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        print("data in backend", data)
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Return the result
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
    app.run(debug=True, host='0.0.0.0', port=3000)
