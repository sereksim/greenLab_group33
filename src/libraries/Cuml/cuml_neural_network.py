import pandas as pd
import cupy as cp
from cuml.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
from pathlib import Path
import os

# Ensure GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

PROJECT_ROOT = Path.cwd()
PATH_DATA = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "Cuml"

def save_measurements(accuracy: float, num_classes: int) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "cuML Neural Network (GPU)",
        "accuracy": accuracy,
        "num_classes": num_classes,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_cumlNN_GPU.csv"
    json_path = RESULTS_PATH / "metrics_cumlNN_GPU.json"

    # Save to CSV
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        pd.concat([old, results], ignore_index=True).to_csv(csv_path, index=False)
    else:
        results.to_csv(csv_path, index=False)

    # Save to JSON
    if json_path.exists():
        old = pd.read_json(json_path)
        pd.concat([old, results], ignore_index=True).to_json(json_path, orient="records", indent=4)
    else:
        results.to_json(json_path, orient="records", indent=4)

def run_cuml_nn_classification():
    data = pd.read_csv(PATH_DATA)
    
    # Detect label column (M/B)
    label_col = None
    for col in data.columns:
        if data[col].astype(str).str.lower().isin(['m', 'b']).any():
            label_col = col
            break
    if label_col is None:
        raise ValueError("No label column containing 'M'/'B' found in dataset.")

    # Convert 'M' -> 1, 'B' -> 0
    data['label'] = data[label_col].apply(lambda x: 1 if str(x).lower() == 'm' else 0)
    X = data.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
    y = data['label']
    
    num_classes = y.nunique()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to GPU arrays
    X_train_gpu = cp.asarray(X_train)
    X_test_gpu = cp.asarray(X_test)
    y_train_gpu = cp.asarray(y_train.values)
    y_test_gpu = cp.asarray(y_test.values)

    # Define cuML Neural Network
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),  # same structure as PyTorch model
        activation='relu',
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_iter=EPOCHS,
        random_state=RANDOM_SEED
    )

    # Train
    model.fit(X_train_gpu, y_train_gpu)

    # Predict
    y_pred_gpu = model.predict(X_test_gpu)
    y_pred = cp.asnumpy(y_pred_gpu)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save results
    save_measurements(accuracy, num_classes)

    print(f"cuML NN Classification (GPU)")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    run_cuml_nn_classification()
