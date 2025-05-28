import mlflow

mlflow.set_tracking_uri("https://dagshub.com/nitinjmv/mlops-pipeline.mlflow")

# Fetch experiment by name
experiment = mlflow.get_experiment_by_name("model_evaluation_experiment")
if experiment is None:
    print("Experiment not found")
else:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    print("Available columns:", runs.columns.tolist())  # Check what exists
    print(runs[['run_id', 'status', 'start_time']])  # Print safely


# best_run = runs.sort_values(by='metrics.accuracy', ascending=False).iloc[0]
# print("Best run ID:", best_run['run_id'])
