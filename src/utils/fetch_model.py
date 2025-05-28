import mlflow

# mlflow.set_tracking_uri("https://dagshub.com/<user>/<repo>.mlflow")

# List all runs from a specific experiment
experiment = mlflow.get_experiment_by_name("model_building_experiment")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Print run IDs
print(runs[['run_id', 'metrics', 'params']])
