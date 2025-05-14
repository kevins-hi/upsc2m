import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------
# Seed for reproducibility
# -----------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Neural Network Definition
# -----------------------
class DifficultyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output range [0, 1]
        )

    def forward(self, x):
        return self.model(x)

# -----------------------
# Evaluation Function
# -----------------------
def evaluate_model(model, dataloader, device, name="Set"):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).squeeze(1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} Evaluation ---")
    print(f"RMSE      : {rmse:.4f}")
    print(f"MAE       : {mae:.4f}")
    print(f"R²        : {r2:.4f}")
    print()
    return y_true, y_pred

# -----------------------
# Main Execution
# -----------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    df = pd.read_csv('../data/features.csv')

    # One-hot encoding
    df = pd.get_dummies(df, columns=['category'])

    # Log-transform skewed columns
    for col in ['obscurity_score_sum', 'named_entities']:
        df[col] = np.log1p(df[col])

    # Split data
    feature_cols = df.columns.difference(['id', 'split', 'difficulty'])
    X_train = df[df['split'] == 'train'][feature_cols]
    y_train = df[df['split'] == 'train']['difficulty']
    X_val = df[df['split'] == 'val'][feature_cols]
    y_val = df[df['split'] == 'val']['difficulty']
    X_test = df[df['split'] == 'test'][feature_cols]
    y_test = df[df['split'] == 'test']['difficulty']

    # Normalize selected features
    ss_cols = [
        'obscurity_score_avg', 'obscurity_score_max', 'obscurity_score_sum',
        'ambiguity_score', 'distractor_quality_avg', 'distractor_quality_max',
        'reading_difficulty', 'negation_presence', 'named_entities',
    ]
    scaler = StandardScaler()
    X_train[ss_cols] = scaler.fit_transform(X_train[ss_cols])
    X_val[ss_cols] = scaler.transform(X_val[ss_cols])
    X_test[ss_cols] = scaler.transform(X_test[ss_cols])

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.astype(np.float32).values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.astype(np.float32).values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.astype(np.float32).values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.astype(np.float32).values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.astype(np.float32).values, dtype=torch.float32)

    # Create datasets and loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

    # Initialize model
    input_dim = X_train_tensor.shape[1]
    model = DifficultyRegressor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss, total_samples = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            total_samples += y_batch.size(0)

        avg_train_loss = total_loss / total_samples

        # Validation
        model.eval()
        total_val_loss, total_val_samples = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch.unsqueeze(1))
                total_val_loss += loss.item() * y_batch.size(0)
                total_val_samples += y_batch.size(0)

        avg_val_loss = total_val_loss / total_val_samples
        print(f"[Epoch {epoch + 1:02d}] Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"\nBest model saved from epoch {best_epoch + 1} with Val Loss = {best_val_loss:.4e}")

    # Load and evaluate best model
    model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=False))
    evaluate_model(model, train_loader, device, name="Train")
    evaluate_model(model, val_loader, device, name="Validation")
    evaluate_model(model, test_loader, device, name="Test")

if __name__ == "__main__":
    main()
