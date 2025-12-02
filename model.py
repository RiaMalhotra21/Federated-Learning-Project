import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# ---------------------------
# 1. Improved Model Definition
# ---------------------------
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------
# 2. Load and Prepare Data (SMOTE Optional)
# ---------------------------
def load_and_prepare_data(file_path, apply_smote=True):
    data = pd.read_csv(file_path)

    # Features used
    features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']

    # Encode type column
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'TRANSFER': 2, 'PAYMENT': 3}
    data['type_num'] = data['type'].map(type_map)
    features.append('type_num')

    # Find missing features
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"⚠️ Warning: Missing features: {missing_features}")
        features = [f for f in features if f in data.columns]

    # Extract X, y
    X = data[features].values
    y = data['isFraud'].values

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---------------------------
    # APPLY SMOTE — ONLY IF True
    # ---------------------------
    if apply_smote:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print(f"✅ SMOTE applied: {sum(y)} frauds / {len(y)} total")
    else:
        print("ℹ️ SMOTE skipped for this dataset (server-side evaluation).")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.shape[1]


# ---------------------------
# 3. Training Function
# ---------------------------
def train_model(model, X_train, y_train, epochs=50, lr=0.0007):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    return model


# ---------------------------
# 4. Evaluation Function
# ---------------------------
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test)
        outputs = model(X_tensor)
        predictions = (outputs >= 0.5).float().squeeze()
        y_tensor = torch.FloatTensor(y_test)
        accuracy = (predictions == y_tensor).float().mean().item()

    print(f"✅ Model Accuracy: {accuracy*100:.2f}%")
    return accuracy
