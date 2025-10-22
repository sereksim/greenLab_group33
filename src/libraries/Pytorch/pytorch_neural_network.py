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
import os
# Set CUDA_VISIBLE_DEVICES to -1 to ensure it uses the CPU, matching the TF script
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RANDOM_SEED = 42
TEST_SIZE = 0.2
EPOCHS = 100 # Kept high to match TF script's large amount of work
BATCH_SIZE = 16 # Changed from 32 to 16 to match TF script
LEARNING_RATE = 0.001

# Set seed for reproducibility
torch.manual_seed(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()

PATH_DATA = PROJECT_ROOT / "data" / "yasserh" / "breast-cancer-dataset" / "versions" / "1" / "breast-cancer.csv"
RESULTS_PATH = PROJECT_ROOT / "src" / "libraries" / "Pytorch"

def save_measurements(accuracy: float, num_classes: int) -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model": "PyTorch NN Classification (CPU)",
        "accuracy": accuracy,
        "num_classes": num_classes, # Will now be 2
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }])

    csv_path = RESULTS_PATH / "metrics_ptNN_CPU.csv"
    json_path = RESULTS_PATH / "metrics_ptNN_CPU.json"

    # Save to CSV
    combined_csv = pd.concat([pd.read_csv(csv_path), results], ignore_index=True) if csv_path.exists() else results
    combined_csv.to_csv(csv_path, index=False)

    # Save to JSON
    combined_json = pd.concat([pd.read_json(json_path), results], ignore_index=True) if json_path.exists() else results
    combined_json.to_json(json_path, orient="records", indent=4)


def run_pytorch_nn_classification():
    data = pd.read_csv(PATH_DATA)
    
    
    label_col = None
    for col in data.columns:
        if data[col].astype(str).str.lower().isin(['m', 'b']).any():
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column containing 'M'/'B' found in dataset.")

    # Convert 'M' (Malignant) to 1 and 'B' (Benign) to 0
    data['label'] = data[label_col].apply(lambda x: 1 if str(x).lower() == 'm' else 0)
    
    # Select features (all numeric columns, dropping the new label and the original one)
    X = data.select_dtypes(include=['number']).drop(columns=['label', label_col], errors='ignore')
    y = data['label']
    
    num_classes = y.nunique() # This will be 2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    # The target is binary (0 or 1), so use float32 and unsqueeze for binary classification
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) 
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define Model, Loss, and Optimizer
    n_features = X_train_scaled.shape[1]
    
    
    model = nn.Sequential(
        nn.Linear(n_features, 32), # Changed from 64 to 32
        nn.ReLU(),
        nn.Linear(32, 16), # Changed from 32 to 16
        nn.ReLU(),
        nn.Linear(16, 1),  # Output layer is 1 for binary classification
        nn.Sigmoid()       # Added Sigmoid for binary output 



    )
    # Use Binary Cross Entropy for binary classification
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train() # Set model to training mode
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
        # Convert probabilities (Sigmoid output) to binary predictions (0 or 1)
        y_pred = (outputs >= 0.5).int() 

    y_test_np = y_test_tensor.int().numpy()
    y_pred_np = y_pred.numpy()
    acc = accuracy_score(y_test_np, y_pred_np)
    
    save_measurements(acc, num_classes)

    print(f"PyTorch NN Classification (CPU)")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    run_pytorch_nn_classification()

