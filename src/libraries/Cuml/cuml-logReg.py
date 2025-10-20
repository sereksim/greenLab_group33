import pandas as pd
import time
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.25
PROJECT_ROOT = Path.cwd()
PATH_CANCER = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"

def save_measurements(acc, prec, rec, f1):
    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "LogisticRegression (cuML GPU)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE
    }])
    out_dir = PROJECT_ROOT / "src" / "libraries" / "cuml"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_logReg.csv"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        pd.concat([old, results], ignore_index=True).to_csv(csv_path, index=False)
    else:
        results.to_csv(csv_path, index=False)

    json_path = out_dir / "metrics_logReg.json"
    if json_path.exists():
        old = pd.read_json(json_path)
        pd.concat([old, results], ignore_index=True).to_json(json_path, orient="records", indent=4)
    else:
        results.to_json(json_path, orient="records", indent=4)

def run_cuml_logistic():
    data = pd.read_csv(PATH_CANCER).drop("id", axis=1).dropna()
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    model = LogisticRegression(max_iter=10000)
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - start

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    save_measurements(acc, prec, rec, f1)

    print(f"✅ cuML LogisticRegression done in {elapsed:.4f}s")
    print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    run_cuml_logistic()
