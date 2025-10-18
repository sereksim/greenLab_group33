import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
PATH_DATA = PROJECT_ROOT / "data" / "camnugent" / "california-housing-prices" / "versions" / "1" / "housing.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "pytorch_linear_regression"

def save_measurements(r2: float, mse: float, rmse: float, mae: float) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "PyTorch Linear Regression (CPU)",
        "r2_score": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_ptLR.csv"
    json_path = RESULTS_PATH / "metrics_ptLR.json"

    # Save to CSV
    combined_csv = pd.concat([pd.read_csv(csv_path), results], ignore_index=True) if csv_path.exists() else results
    combined_csv.to_csv(csv_path, index=False)

    # Save to JSON
    combined_json = pd.concat([pd.read_json(json_path), results], ignore_index=True) if json_path.exists() else results
    combined_json.to_json(json_path, orient="records", indent=4)


def run_pytorch_linear():
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Model, Loss, and Optimizer
    model = nn.Linear(X_train_scaled.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Prediction and Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()

    y_test_np = y_test_tensor.numpy()
    mse = mean_squared_error(y_test_np, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test_np, y_pred)
    r2 = r2_score(y_test_np, y_pred)
    
    save_measurements(r2, mse, rmse, mae)

    print(f"PyTorch Linear Regression (CPU)")
    print(f"R² score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")


if __name__ == "__main__":
    run_pytorch_linear()
