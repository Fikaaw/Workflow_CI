import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import json
from datetime import datetime
# Commented out DagHub to avoid URI conflicts
# from dagshub import dagshub_logger
# import dagshub
# dagshub.init(repo_owner='Fikaaw', repo_name='lung_cancer_modelling_experiment_tracking', mlflow=True)


# Set tracking URI ke local file storage
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
tracking_uri = "file:///" + os.path.join(current_dir, "mlruns").replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)


print(f"MLflow Tracking URI: {tracking_uri}")
print("Untuk melihat MLflow UI, jalankan: mlflow ui --port 5001")
print("Kemudian buka: http://localhost:5001")


mlflow.set_experiment("Pulmonary Cancer Prediction Tuning")


# Load data dan split train-test
data = pd.read_csv("pulmonarycancerclean.csv")


# Convert columns to appropriate data types
for col in data.columns:
    if col != 'pulmonary_cancer':
        data[col] = data[col].astype('float64')
    else:
        # Keep target variable as int for classification
        data[col] = data[col].astype('int')


# Enable autolog for automatic tracking
mlflow.sklearn.autolog()


X_train, X_test, y_train, y_test = train_test_split(
    data.drop("pulmonary_cancer", axis=1),
    data["pulmonary_cancer"],
    test_size=0.2,
    random_state=42
)


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Tuning_{timestamp}"


with mlflow.start_run(run_name=run_name) as run:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])


    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }


    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1)


    grid_search.fit(X_train, y_train)


    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)


    # Logging manual metric dan parameter
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_class_0", report['0']['precision'])
    mlflow.log_metric("recall_class_0", report['0']['recall'])
    mlflow.log_metric("precision_class_1", report['1']['precision'])
    mlflow.log_metric("recall_class_1", report['1']['recall'])


    # Simpan confusion matrix sebagai gambar
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('KNN Confusion Matrix (Tuned)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_path = "training_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)


    # Simpan model
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(best_model, "best_knn_model", signature=signature)


    # Simpan classification_report sebagai JSON
    with open("metric_info.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("metric_info.json")


    # Buat dan simpan estimator.html
    with open("estimator.html", "w") as f:
        f.write("<html><body>")
        f.write("<h1>Best Estimator</h1>")
        f.write(f"<pre>{str(best_model)}</pre>")
        f.write("</body></html>")
    mlflow.log_artifact("estimator.html")