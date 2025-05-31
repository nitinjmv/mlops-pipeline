import datetime

from requests import Response
from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from nltk.stem.porter import PorterStemmer
from time import time
from nltk.corpus import stopwords
import nltk
import string
from src.utils.commons import load_params, logging_setup
from prometheus_client import Counter, Histogram, generate_latest

nltk.download('stopwords')
nltk.download('punkt_tab')

logger = logging_setup('app')

model_path = os.path.join("models", "model", "model.pkl")
logger.debug(f'model_path {model_path}')
with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        start = time()
        message = request.form["message"]
        transformed = transform_text(message)
        pred = model.predict([transformed])
        prediction = "Spam" if pred else "Ham"
        latency = time() - start
        logger.info({
            "timestamp": datetime.now().isoformat(),
            "input": message,
            "prediction": prediction.tolist(),
            "latency": latency
        })
    return render_template("index.html", prediction=prediction)


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

