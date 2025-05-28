# serve_model.py
import joblib
import os
from flask import Flask, request, jsonify

model = joblib.load("path/to/model.pkl")
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["input"]
    prediction = model.predict([data])
    return jsonify(prediction=prediction.tolist())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
