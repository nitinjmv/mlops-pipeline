from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt_tab')

# Load model
model_path = os.path.join("models", "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Flask app
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
        message = request.form["message"]
        transformed = transform_text(message)
        pred = model.predict([transformed])
        prediction = "Spam" if pred else "Ham"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

