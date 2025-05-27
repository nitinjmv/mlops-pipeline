with mlflow.start_run() as run:
    run_id = run.info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
