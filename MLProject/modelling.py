import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import numpy as np
from datetime import datetime
# from dagshub import dagshub_logger
# import dagshub
import os

# Set MLflow tracking URI to local directory to avoid permission issues
# Use absolute path to avoid path resolution issues in GitHub Actions
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
mlruns_path = os.path.join(current_dir, "mlruns")
mlflow.set_tracking_uri(f"file:{mlruns_path}")

# Create a new MLflow Experiment
mlflow.set_experiment("Pulmonary Cancer Prediction")

data = pd.read_csv("pulmonarycancerclean.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("pulmonary_cancer", axis=1),
    data["pulmonary_cancer"],
    random_state=42,
    test_size=0.2
)
input_example = X_train[0:5]

# Check if we're running inside an MLflow project (CLI creates the run)
if mlflow.active_run() is not None:
    # We're inside an MLflow CLI run, don't create nested run
    print(f"Using existing MLflow run: {mlflow.active_run().info.run_id}")
else:
    # We're running standalone, create our own run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"KNN_Modelling_{timestamp}"
    mlflow.start_run(run_name=run_name)

# Log parameters
n_neighbors = 5
algorithm = 'auto'
mlflow.log_param("n_neighbors", n_neighbors)
mlflow.log_param("algorithm", algorithm)

mlflow.autolog(disable=True)  # Nonaktifkan autolog agar tidak bentrok saat log manual

# Train model
model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
model.fit(X_train, y_train)

# Log model
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    input_example=input_example
)

# Predict and log metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mlflow.log_metric("accuracy", accuracy)

# End run if we created it ourselves
if mlflow.active_run() and 'KNN_Modelling_' in str(mlflow.active_run().info.run_name):
    mlflow.end_run()