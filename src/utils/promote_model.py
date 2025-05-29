import sys
import mlflow
import shutil
from mlflow.tracking import MlflowClient
from src.utils.commons import logging_setup
import mlflow.pyfunc

logger = logging_setup('model_promotion')

run_id = sys.argv[1]
logger.debug(f'run_id found {run_id}')

EXPERIMENT_NAME = "model_evaluation_experiment"
REGISTERED_MODEL_NAME = "SpamClassifierModel"

client = MlflowClient()
    
def promote_best_model():
    EVAL_METRIC = "auc"
    METRIC_THRESHOLD = 0.90

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    # --- STEP 1: Find best run above threshold ---
    best_run = None
    best_metric_value = float('-inf')

    print(f"Searching runs in experiment '{EXPERIMENT_NAME}' for highest {EVAL_METRIC} above threshold {METRIC_THRESHOLD}...")

    for run_info in client.search_runs(experiment_ids=[experiment_id], order_by=[f"metrics.{EVAL_METRIC} DESC"]):
        run = run_info.to_dictionary()
        metrics = run.get('data', {}).get('metrics', {})
        metric_val = metrics.get(EVAL_METRIC)

        if metric_val is not None and metric_val >= METRIC_THRESHOLD:
            if metric_val > best_metric_value:
                best_metric_value = metric_val
                best_run = run_info

    if best_run is None:
        print(f"No run found with {EVAL_METRIC} >= {METRIC_THRESHOLD}. Exiting.")
        exit(1)

    print(f"Best run found: {best_run.info.run_id} with {EVAL_METRIC} = {best_metric_value}")

    return best_run.info.run_id


if run_id:
    model_uri = f"runs:/{run_id}/model"    
else:
    model_uri = f"runs:/{promote_best_model()}/model"

registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True
)
print(f"Promoted model version {registered_model.version} to 'Production' stage.")

model = mlflow.pyfunc.load_model(model_uri)
shutil.copytree(model_uri, "models")