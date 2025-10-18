import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "pytorch_nn_classification"

def save_measurements(accuracy: float, num_classes: int) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "PyTorch NN Classification (CPU)",
        "accuracy": accuracy,
        "num_classes": num_classes,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_ptNNC.csv"
    json_path = RESULTS_PATH / "metrics_ptNNC.json"

    # Save to CSV
    combined_csv = pd.concat([pd.read_csv(csv_path), results], ignore_index=True) if csv_path.exists() else results
    combined_csv.to_csv(csv_path, index=False)

    # Save to JSON
    combined_json = pd.concat([pd.read_json(json_path), results], ignore_index=True) if json_path.exists() else results
    combined_json.to_json(json_path, orient="records", indent=4)


def run_pytorch_nn_classification():
    data = pd.read_csv(PATH_DATA)
    data = data.dropna()

    if 'ocean_proximity' in data.columns:
        data = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

    # Create multiclass target: 4 quantiles (0, 1, 2, 3)
    y_multi = pd.qcut(data['median_house_value'], q=4, labels=False, duplicates='drop')
    X = data.drop(columns=['median_house_value'])
    num_classes = y_multi.nunique()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_multi
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long) # CrossEntropyLoss needs long
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Model, Loss, and Optimizer
    n_features = X_train_scaled.shape[1]
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()

    y_test_np = y_test_tensor.numpy()
    acc = accuracy_score(y_test_np, y_pred)
    
    save_measurements(acc, num_classes)

    print(f"PyTorch NN Classification (CPU)")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_pytorch_nn_classification()
