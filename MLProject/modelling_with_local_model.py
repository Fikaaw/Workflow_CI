import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import numpy as np
from datetime import datetime
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

# Opsi 1: Load model yang sudah ada (jika ada)
model_path = "../model_dir/model"  # Path ke model yang sudah ada
use_existing_model = True  # Set ke False jika ingin train model baru

if use_existing_model and os.path.exists(model_path):
    print("Loading existing model...")
    try:
        # Load model yang sudah ada
        model = mlflow.sklearn.load_model(model_path)
        print(f"Model berhasil dimuat dari: {model_path}")
        
        # Evaluate model without MLflow logging to avoid permission issues
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy dengan model yang sudah ada: {accuracy:.4f}")
        print("Model evaluation completed successfully without MLflow logging.")
            
    except Exception as e:
        print(f"Error loading existing model: {str(e)}")
        print("Falling back to training new model...")
        use_existing_model = False

else:
    print("Training new model...")
    use_existing_model = False

# Opsi 2: Train model baru jika model lama tidak ada atau error
if not use_existing_model:
    # Check if we're running inside an MLflow project
    if mlflow.active_run() is not None:
        print(f"Using existing MLflow run: {mlflow.active_run().info.run_id}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"KNN_Training_{timestamp}"
        mlflow.start_run(run_name=run_name)

    # Log parameters
    n_neighbors = 5
    algorithm = 'auto'
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("model_source", "newly_trained")

    mlflow.autolog(disable=True)

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
    
    print(f"Accuracy dengan model baru: {accuracy:.4f}")

    # End run if we created it ourselves
    if mlflow.active_run() and ('KNN_Training_' in str(mlflow.active_run().info.run_name) or 'KNN_Evaluation_' in str(mlflow.active_run().info.run_name)):
        mlflow.end_run()

print("Proses selesai!")