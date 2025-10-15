import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path.cwd()

PATH_HOUSES = PROJECT_ROOT / "data" / "camnugent" / "california-housing-prices" / "versions" / "1" / "housing.csv"

def run_scikit_linear():
    model = LinearRegression()
    data = pd.read_csv(PATH_HOUSES)
    data = data.drop("ocean_proximity", axis=1)
    data = data.dropna(axis=0)
    print(data.head())
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=RANDOM_SEED)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    print(predictions[:5])
    score = model.score(X_test, Y_test)
    print(f"R² score: {score:.3f}")
    mse = mean_squared_error(Y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(Y_test, predictions)

    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")


if __name__ == '__main__':
    run_scikit_linear()
