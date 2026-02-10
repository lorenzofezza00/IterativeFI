import torch
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np


def load_digits_pca(n_components=10, test_size=0.2):
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_loader, test_loader, le

# Esempio di utilizzo
if __name__ == "__main__":
    train_loader, test_loader, le = load_digits_pca()

    print(f"Numero classi: {len(le.classes_)}")
    for batch_x, batch_y in train_loader:
        print("Batch X shape:", batch_x.shape)
        print("Batch y shape:", batch_y.shape)
        break
