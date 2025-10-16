import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32

PROJECT_ROOT = Path.cwd()
PATH_DATA = PROJECT_ROOT / "data" / "camnugent" / "california-housing-prices" / "versions" / "1" / "housing.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "tensorflow"

def save_measurements(r2: float, mse: float, rmse: float, mae: float) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "TensorFlow Linear Regression",
        "r2_score": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_tfReg.csv"
    json_path = RESULTS_PATH / "metrics_tfReg.json"

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


def run_tensorflow_linear():
    data = pd.read_csv(PATH_DATA)
    data = data.dropna()

    if 'ocean_proximity' in data.columns:
        data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

    X = data.drop(columns=['median_house_value'])
    y = data['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='mse'
    )

    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)

    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
   
    save_measurements(r2, mse, rmse, mae)

    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    run_tensorflow_linear()
