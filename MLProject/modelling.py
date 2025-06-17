import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import mlflow.sklearn
from datetime import datetime
import os
import argparse

def setup_mlflow():
    """Setup MLflow tracking URI and experiment"""
    # Set MLflow tracking URI to local directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mlruns_path = os.path.join(current_dir, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

def load_and_prepare_data():
    """Load and prepare the dataset"""
    data = pd.read_csv("pulmonarycancerclean.csv")
    
    for col in data.columns:
        if col != 'pulmonary_cancer':
            data[col] = data[col].astype('float64')
        else:
            # Keep target variable as int for classification
            data[col] = data[col].astype('int')
    
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("pulmonary_cancer", axis=1),
        data["pulmonary_cancer"],
        test_size=0.2,
        random_state=42
    )
    
    return data, X_train, X_test, y_train, y_test

def run_base_model(X_train, X_test, y_train, y_test):
    """Run base KNN model without hyperparameter tuning"""
    # Check for active run, if not found, start a new one
    active_run = mlflow.active_run()
    if active_run:
        print(f"Using active MLflow run: {active_run.info.run_id}")
    else:
        print("No active run found, starting new run...")
        mlflow.start_run(run_name="KNN_Base_Modelling")
        active_run = mlflow.active_run()
        print(f"Started new MLflow run: {active_run.info.run_id}")
    
    input_example = X_train[0:5]
    
    # Set parameters
    n_neighbors = 5
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", "auto")
    mlflow.log_param("weights", "uniform")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("model_type", "base")
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    model.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example
    )
    
    # Calculate and log metrics
    accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("data_size", len(X_train) + len(X_test))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    
    print(f"Base model trained with test accuracy: {accuracy:.4f}")
    print(f"Base model train accuracy: {train_accuracy:.4f}")
    print("Base model MLflow run completed successfully")
    
    return model, accuracy

def run_tuning_model(X_train, X_test, y_train, y_test):
    """Run hyperparameter tuning for KNN model"""
    # Check for active run, if not found, start a new one
    active_run = mlflow.active_run()
    if active_run:
        print(f"Using active MLflow run: {active_run.info.run_id}")
    else:
        print("No active run found, starting new run...")
        mlflow.start_run(run_name="KNN_Tuning_Modelling")
        active_run = mlflow.active_run()
        print(f"Started new MLflow run: {active_run.info.run_id}")
    
    # Log model type
    mlflow.log_param("model_type", "tuned")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Define parameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Log hyperparameter search space
    mlflow.log_param("param_grid", str(param_grid))
    mlflow.log_param("cv_folds", 5)
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid=param_grid,
        cv=5, 
        scoring='accuracy', 
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model and predictions
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = best_model.score(X_train, y_train)
    
    # Log best parameters
    for param, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{param}", value)
    
    # Log model
    input_example = X_train[0:5]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        input_example=input_example
    )
    
    # Log metrics
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("data_size", len(X_train) + len(X_test))
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size", len(X_test))
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print("Tuning model MLflow run completed successfully")
    
    return best_model, test_accuracy

def main():
    """Main function to run both base and tuning models"""
    parser = argparse.ArgumentParser(description='Run KNN modeling with optional tuning')
    parser.add_argument('--mode', choices=['base', 'tuning', 'both'], default='both',
                       help='Mode to run: base model only, tuning only, or both')
    
    args = parser.parse_args()
    
    # Setup MLflow
    setup_mlflow()
    
    # Load and prepare data
    print("Loading and preparing data...")
    data, X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Data loaded: {len(data)} samples, {X_train.shape[1]} features")
    
    # Run models based on mode
    if args.mode in ['base', 'both']:
        print("\n=== Running Base Model ===")
        base_model, base_accuracy = run_base_model(X_train, X_test, y_train, y_test)
    
    if args.mode in ['tuning', 'both']:
        print("\n=== Running Hyperparameter Tuning ===")
        tuned_model, tuned_accuracy = run_tuning_model(X_train, X_test, y_train, y_test)
    
    # Compare results if both models were run
    if args.mode == 'both':
        print("\n=== Model Comparison ===")
        print(f"Base model accuracy: {base_accuracy:.4f}")
        print(f"Tuned model accuracy: {tuned_accuracy:.4f}")
        improvement = tuned_accuracy - base_accuracy
        print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    print("\nAll modeling completed successfully!")

if __name__ == "__main__":
    main()