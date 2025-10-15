import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from sklearnex import patch_sklearn
patch_sklearn()

def load_dataset(path):
    df = pd.read_csv(path)
    if 'median_house_value' not in df.columns:
        raise ValueError("Dataset must contain 'median_house_value' column.")
    y = df['median_house_value']
    X = df.drop(columns=['median_house_value'])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y

def run_daal4py_linReg(csv_path):
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression(n_jobs=-1)

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    execution_time = end_time - start_time

    print("\n=== Linear Regression (Intel oneDAL / daal4py) ===")
    print(f"RMSE         : {rmse:.4f}")
    print(f"R² Score     : {r2:.4f}")
    print(f"Time Taken   : {execution_time:.4f} seconds")

if __name__ == "__main__":

    run_daal4py_linReg("data/housing.csv")
