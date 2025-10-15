import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path.cwd()

PATH_CANCER = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"


def run_scikit_linear():
    model = LogisticRegression(max_iter=10000, random_state=RANDOM_SEED)
    data = pd.read_csv(PATH_CANCER)
    data = data.drop('id', axis=1)
    data = data.dropna(axis=0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    print(data.info())
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=RANDOM_SEED)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(predictions[:5])
    accuracy = accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)

    print(f"\nAccuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")


if __name__ == '__main__':
    run_scikit_linear()
