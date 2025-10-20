import pandas as pd
import time
from cuml.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.25
PROJECT_ROOT = Path.cwd()
PATH_HOUSES = PROJECT_ROOT / "data" / "camnugent" / "california-housing-prices" / "versions" / "1" / "housing.csv"

def save_measurements(r_score, mse, rmse):
    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "LinearRegression (cuML GPU)",
        "r2_score": r_score,
        "MSE": mse,
        "RMSE": rmse,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE
    }])

    out_dir = PROJECT_ROOT / "src" / "libraries" / "cuml"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics_linReg.csv"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        pd.concat([old, results], ignore_index=True).to_csv(csv_path, index=False)
    else:
        results.to_csv(csv_path, index=False)

    json_path = out_dir / "metrics_linReg.json"
    if json_path.exists():
        old = pd.read_json(json_path)
        pd.concat([old, results], ignore_index=True).to_json(json_path, orient="records", indent=4)
    else:
        results.to_json(json_path, orient="records", indent=4)

def run_cuml_linear():
    data = pd.read_csv(PATH_HOUSES).drop("ocean_proximity", axis=1).dropna()
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    model = LinearRegression(fit_intercept=True)
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.time() - start

    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5

    save_measurements(r2, mse, rmse)

    print(f"✅ cuML LinearRegression done in {elapsed:.4f}s")
    print(f"R²: {r2:.3f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    run_cuml_linear()
