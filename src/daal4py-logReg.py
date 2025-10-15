
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearnex import patch_sklearn
patch_sklearn()

def load_dataset(path):
    df = pd.read_csv(path)
    if 'diagnosis' in df.columns:
        y = df['diagnosis'].map({'M': 1, 'B': 0})
        X = df.drop(columns=['diagnosis'])
    elif 'target' in df.columns:
        y = df['target']
        X = df.drop(columns=['target'])
    else:
        raise ValueError("Dataset must contain 'diagnosis' or 'target' column.")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y

def run_daal4py_logReg(csv_path):
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    execution_time = end_time - start_time

    print("\n=== Logistic Regression (Intel oneDAL / daal4py) ===")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Time Taken   : {execution_time:.4f} seconds")

if __name__ == "__main__":
    run_daal4py_logReg("data/breast-cancer.csv")
