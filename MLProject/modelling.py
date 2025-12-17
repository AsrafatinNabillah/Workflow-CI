import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn

def train_model(data_path: str):
    # ===============================
    # 1. LOAD DATA PREPROCESSING
    # ===============================
    df = pd.read_excel(data_path)

    feature_cols = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    target_col = 'ISPU_Kategori'

    X = df[feature_cols]
    y = df[target_col]

    # ===============================
    # 2. TRAIN TEST SPLIT
    # ===============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ISPU_RandomForest")
    mlflow.sklearn.autolog()

    # ===============================
    # 3. TRAIN MODEL
    # ===============================
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    # ===============================
    # 4. EVALUATION
    # ===============================
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ===============================
    # 5. SAVE MODEL
    # ===============================
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "random_forest_ispu.pkl"
    joblib.dump(model, model_path)

    print(f"\nModel disimpan di: {model_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "DataISPU_prepocessing.xlsx"

    train_model(str(DATA_PATH))
