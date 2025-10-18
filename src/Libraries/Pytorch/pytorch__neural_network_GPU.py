import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
from pathlib import Path

RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
PATH_DATA = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "pytorch"

# GPU config
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Exiting program.")
    sys.exit("Cannot proceed without GPU.")

def save_measurements(accuracy: float, precision: float, recall: float, f1: float) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "PyTorch Neural Network GPU",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_ptNN_GPU.csv"
    json_path = RESULTS_PATH / "metrics_ptNN_GPU.json"

    # Save to CSV
    combined_csv = pd.concat([pd.read_csv(csv_path), results], ignore_index=True) if csv_path.exists() else results
    combined_csv.to_csv(csv_path, index=False)

    # Save to JSON
    combined_json = pd.concat([pd.read_json(json_path), results], ignore_index=True) if json_path.exists() else results
    combined_json.to_json(json_path, orient="records", indent=4)


def run_pytorch_nn_GPU():
    data = pd.read_csv(PATH_DATA)

    # Find the label column (M/B)
    label_col = None
    for col in data.columns:
        if data[col].astype(str).str.lower().isin(['m', 'b']).any():
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column containing 'M'/'B' found in dataset.")

    # Convert label to 1 (M) or 0 (B)
    data['label'] = data[label_col].apply(lambda x: 1 if str(x).lower() == 'm' else 0)
    
    # Select only numeric features
    X = data.select_dtypes(include=['number']).drop(columns=['label'], errors='ignore')
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors and move to GPU
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    # Use float32 and .view(-1, 1) for BCEWithLogitsLoss
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Model (matches TF structure: 32 -> 16 -> 1)
    n_features = X_train_scaled.shape[1]
    model = nn.Sequential(
        nn.Linear(n_features, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)  # No sigmoid, as BCEWithLogitsLoss includes it
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        # Apply sigmoid to logits to get probabilities, then threshold
        y_pred = (torch.sigmoid(y_pred_logits) >= 0.5).int().cpu().numpy()

    y_test_np = y_test_tensor.int().cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test_np, y_pred)
    precision = precision_score(y_test_np, y_pred, zero_division=0)
    recall = recall_score(y_test_np, y_pred, zero_division=0)
    f1 = f1_score(y_test_np, y_pred, zero_division=0)

    save_measurements(accuracy, precision, recall, f1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")


if __name__ == "__main__":
    run_pytorch_nn_GPU()
