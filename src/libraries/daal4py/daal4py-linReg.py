import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from pathlib import Path
from sklearnex import patch_sklearn

# Enable Intel oneDAL (daal4py) acceleration
patch_sklearn()

RANDOM_SEED = 42
TEST_SIZE = 0.25
PROJECT_ROOT = Path.cwd()
PATH_HOUSES = PROJECT_ROOT / "data" / "camnugent" / "california-housing-prices" / "versions" / "1" / "housing.csv"

def save_measurements(r_score: float, mse: float, rmse: float, mae: float) -> None:
    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "LinearRegression (Intel oneDAL)",
        "r2_score": r_score,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE
    }])

    results_path = PROJECT_ROOT / "src" / "libraries" / "daal4py"
    results_path.mkdir(parents=True, exist_ok=True)

    results_path_csv = results_path / "metrics_linReg.csv"
    if results_path_csv.exists():
        old = pd.read_csv(results_path_csv)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_csv(results_path_csv, index=False)

    results_path_json = results_path / "metrics_linReg.json"
    if results_path_json.exists():
        old = pd.read_json(results_path_json)
        combined = pd.concat([old, results], ignore_index=True)
    else:
        combined = results
    combined.to_json(results_path_json, orient="records", indent=4)

def run_daal4py_linear():
    model = LinearRegression(n_jobs=-1)
    data = pd.read_csv(PATH_HOUSES)
    data = data.drop("ocean_proximity", axis=1)
    data = data.dropna(axis=0)
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(predictions[:5])

    r_score = model.score(X_test, Y_test)
    mse = mean_squared_error(Y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(Y_test, predictions)

    save_measurements(r_score, mse, rmse, mae)

    print(f"R² score: {r_score:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

if __name__ == '__main__':
    run_daal4py_linear()
