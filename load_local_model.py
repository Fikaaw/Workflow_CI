import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Load model dari direktori lokal
# Ganti path ini sesuai dengan lokasi model Anda
model_path = "./model_dir/model"  # Path relatif ke model yang sudah ada
# Atau gunakan path absolut:
# model_path = "c:/Users/immab/Documents/workflow-cek/model_dir/model"

print("Loading model dari path lokal...")
try:
    # Load model menggunakan mlflow
    model = mlflow.sklearn.load_model(model_path)
    print(f"Model berhasil dimuat dari: {model_path}")
    print(f"Model type: {type(model)}")
    
    # Load data untuk testing (opsional)
    data = pd.read_csv("./MLProject/lung_cancer_cln.csv")
    
    # Prepare test data
    X_test = data.drop("lung_cancer", axis=1).head(10)  # Ambil 10 data pertama untuk test
    y_test = data["lung_cancer"].head(10)
    
    # Prediksi menggunakan model yang sudah dimuat
    predictions = model.predict(X_test)
    
    print("\nHasil prediksi:")
    for i, (pred, actual) in enumerate(zip(predictions, y_test)):
        print(f"Data {i+1}: Prediksi = {pred}, Aktual = {actual}")
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAkurasi model: {accuracy:.4f}")
    
except Exception as e:
    print(f"Error saat loading model: {str(e)}")
    print("\nPastikan:")
    print("1. Path model sudah benar")
    print("2. Model file ada di lokasi yang ditentukan")
    print("3. MLflow sudah terinstall dengan benar")