import pandas as pd
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datetime import datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 16

PROJECT_ROOT = Path.cwd()
PATH_DATA = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "tensorflow"

def save_measurements(accuracy: float, precision: float, recall: float, f1: float) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "TensorFlow Neural Network CPU",
        "accuracy": accuracy,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_tfNN_CPU.csv"
    json_path = RESULTS_PATH / "metrics_tfNN_CPU.json"

    if csv_path.exists():
        old = pd.read_csv(csv_path)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(csv_path, index=False)

    if json_path.exists():
        old = pd.read_json(json_path)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_json(json_path, orient="records", indent=4)


def run_tensorflow_nn_CPU():
    data = pd.read_csv(PATH_DATA)

    label_col = None
    for col in data.columns:
        if data[col].astype(str).str.lower().isin(['m', 'b']).any():
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column containing 'M'/'B' found in dataset.")

    data['label'] = data[label_col].apply(lambda x: 1 if str(x).lower() == 'm' else 0)

    X = data.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    save_measurements(accuracy, precision, recall, f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")


if __name__ == "__main__":
    run_tensorflow_nn_CPU()
