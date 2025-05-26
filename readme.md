# MLOps Monorepo Pipeline

This project is a monorepo setup for an end-to-end MLOps pipeline using:
- GitHub Actions for CI/CD (with self-hosted runner)
- MLflow for experiment tracking and model versioning
- Flask for model serving
- Streamlit for UI
- Scikit-learn on the Iris dataset

## Usage

```bash
bash mlflow_server.sh  # start MLflow UI on http://localhost:5000
python src/data_pipeline.py
python src/preprocess.py
python src/train.py
pytest src/test_train.py src/test_preprocess.py
python app/app.py      # start Flask server on port 8000
streamlit run app/ui.py  # UI for predictions
```

## CI/CD
Trigger on `push` to main, running pipeline end-to-end