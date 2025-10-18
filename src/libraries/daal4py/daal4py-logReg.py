from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
from sklearnex import patch_sklearn

patch_sklearn()

RANDOM_SEED = 42
TEST_SIZE = 0.25
PROJECT_ROOT = Path.cwd()
PATH_CANCER = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"


def save_measurements(accuracy: float, precision: float, recall: float, f1: float) -> None:
    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "LogisticRegression (Intel oneDAL)",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE
    }])

    results_path = PROJECT_ROOT / "src" / "libraries" / "daal4py"
    results_path.mkdir(parents=True, exist_ok=True)

    results_path_csv = results_path / "metrics_logReg.csv"
    if results_path_csv.exists():
        old = pd.read_csv(results_path_csv)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(results_path_csv, index=False)

    results_path_json = results_path / "metrics_logReg.json"
    if results_path_json.exists():
        old = pd.read_json(results_path_json)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_json(results_path_json, orient="records", indent=4)


def run_daal4py_logistic():
    model = LogisticRegression(max_iter=10000, n_jobs=-1, random_state=RANDOM_SEED)
    data = pd.read_csv(PATH_CANCER)
    data = data.drop('id', axis=1)
    data = data.dropna(axis=0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(predictions[:5])

    accuracy = accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)

    save_measurements(accuracy, precision, recall, f1)

    print("\n=== Logistic Regression (Intel oneDAL / daal4py) ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")


if __name__ == '__main__':
    run_daal4py_logistic()
