FROM python:alpine3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Install Python dependencies
RUN pip install --no-cache-dir dvc[s3] pandas scikit-learn mlflow

# Clone repo and pull model
RUN git clone https://dagshub.com/nitinjmv/mlops-pipeline.git . \
    && dvc pull models/model.pkl.dvc \
    && ls -lah

COPY . .

CMD ["python", "app/app.py"]
