FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y git && pip install --upgrade pip

RUN pip install pandas scikit-learn dvc mlflow nltk joblib flask

RUN python -m nltk.downloader punkt stopwords

COPY . .

CMD ["python", "app.py"]
